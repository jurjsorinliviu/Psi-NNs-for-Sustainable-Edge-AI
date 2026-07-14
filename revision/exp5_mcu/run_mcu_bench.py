"""
run_mcu_bench.py -- measured embedded results for the structured Psi-NN and its baselines.

Answers Reviewer 2 ("the paper estimates operations, memory and power from model
structure and vendor specifications, but does not deploy the model on the target MCU or
measure actual latency, RAM/flash use, inference energy, or numerical accuracy") and
Reviewer 1 ("consider whether the proposed operation-count model could be transformed by
compiler optimizations and weight reuse").

What is MEASURED (not estimated):
  * flash and RAM      -- from the linked binary (arm-none-eabi-size). The link uses the
                          REAL target memory map, so an over-large model fails to link.
  * executed instructions -- SysTick under QEMU `-icount shift=0`, where virtual time
                          advances exactly 1 ns per executed instruction. A calibration
                          loop of known instruction count fixes the ticks->instructions
                          factor (measured: 40 instructions/tick on the MPS2 boards).
  * numerical accuracy -- the device's own fp32/int8 outputs are printed and scored on
                          the host against a float64 reference. The antisymmetry residual
                          is recomputed from device output, so the symmetry claim is
                          verified ON TARGET.

What is DERIVED (and labelled as such in the paper):
  * latency  = instructions * CPI / f_clk        (CPI documented per core)
  * energy   = latency * datasheet current * V   (vendor active-mode figures)

This is emulation, not silicon: QEMU counts instructions, it does not model the pipeline,
flash wait states, or caches. The paper must say so.

Usage:
    python run_mcu_bench.py --targets nrf52840 stm32h7 --opts O2
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import BurgersObjective, RESULTS, Scorer, dense_pinn, device_of, psinn, train_pinn  # noqa: E402
from export_c import export_mlp, export_psinn_burgers  # noqa: E402
from distill_cluster import ClusteredModel, distill_student, retrain_centroids  # noqa: E402

INSTR_PER_TICK = 40          # calibrated: 200,001 instructions -> 5,000 SysTick ticks

TARGETS = {
    # name        qemu machine   cpu          mcpu/fpu flags                     flash  ram   f_MHz  CPI  uA/MHz  V
    "nrf52840": dict(machine="mps2-an386", cpu="cortex-m4",
                     flags=["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"],
                     flash_kb=1024, ram_kb=256, f_mhz=64, cpi=1.25, ua_per_mhz=52.0, volt=3.0),
    "stm32h7":  dict(machine="mps2-an500", cpu="cortex-m7",
                     flags=["-mcpu=cortex-m7", "-mfpu=fpv5-d16", "-mfloat-abi=hard"],
                     flash_kb=2048, ram_kb=1024, f_mhz=480, cpi=1.10, ua_per_mhz=275.0, volt=3.3),
}


def tool(name):
    base = Path(r"C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi")
    for d in sorted(base.glob("*"), reverse=True):
        exe = d / "bin" / f"arm-none-eabi-{name}.exe"
        if exe.exists():
            return str(exe)
    found = shutil.which(f"arm-none-eabi-{name}")
    if not found:
        sys.exit(f"arm-none-eabi-{name} not found")
    return found


QEMU = shutil.which("qemu-system-arm") or r"C:\Program Files\qemu\qemu-system-arm.exe"


# --------------------------------------------------------------------------
def make_test_set(n=128):
    """Mirrored (t, x) / (t, -x) pairs so the antisymmetry residual can be recomputed
    from the device's own outputs."""
    rng = np.random.default_rng(7)
    t = rng.uniform(0, 1, n // 2).astype(np.float32)
    x = rng.uniform(0, 1, n // 2).astype(np.float32)
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0::2, 0] = t; pts[0::2, 1] = x
    pts[1::2, 0] = t; pts[1::2, 1] = -x
    return pts


def write_test_data(path, pts):
    rows = ",\n    ".join(f"{{{a:.8e}f, {b:.8e}f}}" for a, b in pts)
    path.write_text(f"#define NTEST {len(pts)}\n"
                    f"const float test_in[NTEST][2] = {{\n    {rows}\n}};\n")


def build(target, model_c, model_h, workdir, opt="O2"):
    t = TARGETS[target]
    ld = workdir / "target.ld"
    ld.write_text((HERE / "firmware" / "target.ld.in").read_text()
                  .replace("@FLASH_KB@", str(t["flash_kb"]))
                  .replace("@RAM_KB@", str(t["ram_kb"])))
    (workdir / "model.h").write_text(model_h)
    elf = workdir / f"{target}_{opt}.elf"
    cmd = [tool("gcc"), *t["flags"], "-mthumb", f"-{opt}", "-ffreestanding", "-nostdlib",
           "-fno-math-errno", "-I", str(workdir), "-T", str(ld), "-o", str(elf),
           str(HERE / "firmware" / "main.c"), str(model_c), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return elf, None, None
    # Distinguish "the model genuinely does not fit the target" from "the build is
    # broken". Conflating them would let a toolchain bug masquerade as a hardware
    # result, which is exactly the kind of claim this experiment exists to avoid.
    err = r.stderr.strip()
    overflow = re.search(r"region `(\w+)' overflowed by (\d+) bytes", err)
    if overflow:
        return None, "does_not_fit", {"region": overflow.group(1),
                                      "overflow_bytes": int(overflow.group(2))}
    return None, "build_error", err[-600:]


def size_of(elf):
    out = subprocess.run([tool("size"), str(elf)], capture_output=True, text=True).stdout
    nums = re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)", out, re.M)
    text, data, bss = map(int, nums[0])
    return {"flash_bytes": text + data, "ram_bytes": data + bss,
            "text": text, "data": data, "bss": bss}


def run_qemu(elf, machine, cpu, timeout=180):
    """The firmware spins forever after printing (as real firmware does), so we stream
    QEMU's console and stop it once the guest signals DONE."""
    import time
    cmd = [QEMU, "-machine", machine, "-cpu", cpu, "-kernel", str(elf),
           "-icount", "shift=0", "-nographic", "-serial", "mon:stdio", "-no-reboot"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1)
    lines, deadline = [], time.time() + timeout
    try:
        for line in p.stdout:
            lines.append(line)
            if "DONE" in line or time.time() > deadline:
                break
    finally:
        p.kill()
        p.wait()

    txt = "".join(lines)
    m = re.search(r"TICKS (\d+)", txt)
    if not m or "DONE" not in txt:
        return None, txt[-400:] or "(no output)"
    ticks = int(m.group(1))
    outs = [int(v) / 1e6 for v in re.findall(r"^(-?\d+)$", txt.split("OUT")[1], re.M)]
    return {"ticks": ticks, "instructions": ticks * INSTR_PER_TICK,
            "outputs": np.array(outs, dtype=np.float64)}, None


def score_device(outputs, pts, ref_fn):
    """Relative L2 and antisymmetry residual computed from DEVICE outputs."""
    ref = ref_fn(pts)
    rel = float(np.linalg.norm(outputs - ref) / np.linalg.norm(ref) * 100.0)
    a, b = outputs[0::2], outputs[1::2]          # (t, x) and (t, -x)
    anti = float(np.mean(np.abs(a + b)) / (np.mean(np.abs(a)) + 1e-12))
    return {"rel_l2_pct": rel, "antisymmetry": anti}


def derive(instr, target):
    t = TARGETS[target]
    cycles = instr * t["cpi"]
    latency_s = cycles / (t["f_mhz"] * 1e6)
    current_a = t["ua_per_mhz"] * 1e-6 * t["f_mhz"]
    energy_j = latency_s * current_a * t["volt"]
    return {"cycles_est": cycles, "latency_ms": latency_s * 1e3,
            "energy_uj": energy_j * 1e6,
            "assumptions": {"cpi": t["cpi"], "f_mhz": t["f_mhz"],
                            "ua_per_mhz": t["ua_per_mhz"], "volt": t["volt"]}}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["nrf52840", "stm32h7"])
    ap.add_argument("--opts", nargs="+", default=["O0", "O2", "O3"])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster-eps", type=float, default=0.1)
    args = ap.parse_args()

    dev = device_of("cpu")
    scorer = Scorer(dev)
    obj = BurgersObjective(dev, seed=0)
    pts = make_test_set(128)

    ref_scorer = Scorer(dev)
    ref_fn = lambda p: ref_scorer._fd_ref(torch.tensor(p)).cpu().numpy().ravel().astype(np.float64)

    workdir = RESULTS / "exp5_mcu" / "build"
    workdir.mkdir(parents=True, exist_ok=True)
    write_test_data(workdir / "test_data.h", pts)

    print(f"[exp5] training models (seed {args.seed}, {args.epochs} epochs)")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    m_psinn = train_pinn(psinn().to(dev), obj, epochs=args.epochs)
    torch.manual_seed(args.seed)
    m_dense = train_pinn(dense_pinn().to(dev), obj, epochs=args.epochs)

    # The compressed artifact we ship is the CENTROID-RETRAINED clustered model, i.e.
    # cluster the trained network and then re-optimize the K centroids with the relation
    # structure frozen. exp4 shows that retraining is the stage that preserves accuracy
    # (post-hoc clustering alone destroys the model).
    #
    # Distillation is deliberately NOT used here. It helps only when the teacher is
    # better than the student -- true for the memristor, false on Burgers, where the
    # structured Psi-NN (4% rel-L2) beats the dense teacher (42%) tenfold, so distilling
    # toward that teacher drags the student down (measured: 32% vs 4%).
    print("  clustering + retraining centroids ...")
    torch.manual_seed(args.seed)
    m_clust = ClusteredModel(m_psinn, args.cluster_eps).to(dev)
    retrain_centroids(m_clust, lambda m: obj(m), epochs=args.epochs, lr=1e-3)

    host = {"psinn": scorer.score(m_psinn), "dense": scorer.score(m_dense),
            "psinn_clustered_distilled": {**scorer.score(m_clust),
                                          "clusters": m_clust.n_clusters()}}
    print(f"  host: psinn relL2={host['psinn']['rel_l2_pct']:.1f}% "
          f"antisym={host['psinn']['antisymmetry']:.4f} | "
          f"dense relL2={host['dense']['rel_l2_pct']:.1f}% | "
          f"clustered(K={m_clust.n_clusters()}) "
          f"relL2={host['psinn_clustered_distilled']['rel_l2_pct']:.1f}%")

    variants = [
        ("psinn_f32",   lambda p: export_psinn_burgers(m_psinn, p, "float32"),
         "psinn", 1),
        ("psinn_int8",  lambda p: export_psinn_burgers(m_psinn, p, "int8"),
         "psinn", 1),
        ("psinn_clust_distilled",
         lambda p: export_psinn_burgers(None, p, "clustered", clustered_model=m_clust),
         "psinn", 1),
        ("dense_f32",   lambda p: export_mlp(m_dense, p, "float32", name="model"),
         "model", 1),
        ("dense_int8",  lambda p: export_mlp(m_dense, p, "int8", name="model"),
         "model", 1),
    ]

    results = {"host_reference": host, "instr_per_tick": INSTR_PER_TICK, "variants": {}}
    for vname, exporter, cname, nout in variants:
        cfile = workdir / f"{vname}.c"
        meta = exporter(cfile)
        header = (f"#define MODEL_FORWARD {cname}_forward\n"
                  f"#define MODEL_NOUT {nout}\n"
                  f"void {cname}_forward(const float *in, float *out);\n")
        print(f"\n  {vname}: {meta['params']} params, "
              f"{meta['weight_bytes']} B weights"
              + (f", K={meta['clusters']}" if meta.get("clusters") else ""))
        entry = {"export": meta, "targets": {}}

        for tgt in args.targets:
            for opt in args.opts:
                elf, kind, detail = build(tgt, cfile, header, workdir, opt)
                key = f"{tgt}/{opt}"
                if elf is None and kind == "does_not_fit":
                    entry["targets"][key] = {"link": "does_not_fit", **detail}
                    print(f"    {key:<20} DOES NOT FIT: {detail['region']} overflowed "
                          f"by {detail['overflow_bytes']} B")
                    continue
                if elf is None:
                    entry["targets"][key] = {"link": "build_error", "error": detail}
                    print(f"    {key:<20} BUILD ERROR (not a fit result):\n{detail}")
                    continue
                sz = size_of(elf)
                run, rerr = run_qemu(elf, TARGETS[tgt]["machine"], TARGETS[tgt]["cpu"])
                if run is None:
                    entry["targets"][key] = {"link": "ok", **sz, "run": "FAILED",
                                             "error": rerr}
                    print(f"    {key:<20} flash={sz['flash_bytes']:>6}B "
                          f"ram={sz['ram_bytes']:>6}B  RUN FAILED")
                    continue
                acc = score_device(run["outputs"], pts, ref_fn)
                der = derive(run["instructions"], tgt)
                per_inf = run["instructions"] / len(pts)
                entry["targets"][key] = {
                    "link": "ok", **sz,
                    "instructions_total": run["instructions"],
                    "instructions_per_inference": per_inf,
                    "device_accuracy": acc,
                    "derived": {k: (v / len(pts) if k in ("latency_ms", "energy_uj", "cycles_est")
                                    else v) for k, v in der.items()},
                }
                print(f"    {key:<20} flash={sz['flash_bytes']:>6}B ram={sz['ram_bytes']:>6}B "
                      f"instr/inf={per_inf:>8.0f}  relL2={acc['rel_l2_pct']:5.1f}% "
                      f"antisym={acc['antisymmetry']:.4f}")
        results["variants"][vname] = entry

    outdir = RESULTS / "exp5_mcu"
    (outdir / "mcu_results.json").write_text(json.dumps(results, indent=2, default=float),
                                             encoding="utf-8")
    print(f"\n[exp5] wrote {outdir / 'mcu_results.json'}")


if __name__ == "__main__":
    main()
