"""
exp7_simulator_scaling.py -- does the structured export's speed advantage survive scale?

Answers Reviewer 1: "Considering larger circuits and more complex situations, is the
reported simulator speed increase of ~10% still substantial?"

The submitted comparison used a single device, where solver overhead is shared by both
models and therefore dilutes any model-evaluation advantage. The reviewer's question is
whether the margin grows, shrinks, or vanishes as the circuit grows. We answer it by
sweeping the number of device instances and re-measuring.

Circuit: N memristor instances driven in parallel from one source (a crossbar column).
As N grows, the fraction of simulator time spent inside per-device model evaluation grows
with it, so the dense-vs-structured gap should widen if the advantage is real.

Both models are compiled with OpenVAF to OSDI and run in ngspice; wall time is the
minimum of `--reps` runs, which is the standard way to suppress OS scheduling noise.

Usage:
    python exp7_simulator_scaling.py --sizes 1 4 16 64 256 --reps 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS  # noqa: E402

WD = Path("C:/ngspice_work")
OPENVAF = WD / "openvaf" / "openvaf.exe"
NGSPICE = "ngspice"
VA_DIR = (HERE.parent / "Psi-HDL-implementation" / "Code" / "spice_cost_validation")
MODELS = {"dense": "mem_dense", "structured": "mem_struct"}

TESTBENCHES = {
    "dc":    ("V1 a 0 0",                             "dc V1 -1 1 0.002"),
    "pulse": ("V1 a 0 PULSE(-1 1 0 1u 1u 20u 50u)",   "tran 0.05u 2m uic"),
    "sine":  ("V1 a 0 SIN(0 1 1k)",                   "tran 0.05u 2m uic"),
}


def compile_osdi(model):
    """OpenVAF: .va -> .osdi (once per model)."""
    va = VA_DIR / f"{MODELS[model]}.va"
    osdi = WD / f"{MODELS[model]}.osdi"
    if osdi.exists():
        return osdi
    r = subprocess.run([str(OPENVAF), str(va), "-o", str(osdi)],
                       capture_output=True, text=True, cwd=str(WD))
    if r.returncode != 0 or not osdi.exists():
        sys.exit(f"OpenVAF failed for {model}:\n{r.stdout}\n{r.stderr}")
    print(f"  compiled {osdi.name}")
    return osdi


def netlist(model, n_devices, tb):
    src, analysis = TESTBENCHES[tb]
    osdi = (WD / f"{MODELS[model]}.osdi").as_posix()
    out = (WD / f"scale_{model}_{tb}_{n_devices}.txt").as_posix()
    # N instances in parallel between the driven node and ground, each through its own
    # series resistor so the devices do not collapse into one trivially-shared solve.
    inst = "\n".join(
        f"R{i} a n{i} 100\nN{i} n{i} 0 dut" for i in range(n_devices))
    code = f"""* scaling {model} N={n_devices} {tb}
{src}
{inst}
.model dut {MODELS[model]}()
.control
pre_osdi {osdi}
{analysis}
wrdata {out} i(v1)
quit
.endc
.end
"""
    return code, out


def run_once(model, n, tb, reps):
    code, out = netlist(model, n, tb)
    cf = WD / f"scale_{model}_{tb}_{n}.cir"
    cf.write_text(code)
    best, pts, ok = None, 0, True
    for _ in range(reps):
        if os.path.exists(out):
            os.remove(out)
        t0 = time.perf_counter()
        r = subprocess.run([NGSPICE, "-b", str(cf)], capture_output=True, text=True,
                           timeout=900)
        dt = time.perf_counter() - t0
        log = (r.stdout + r.stderr).lower()
        if r.returncode != 0 or any(k in log for k in
                                    ("timestep too small", "singular", "no convergence",
                                     "fatal", "aborted")):
            ok = False
            break
        if os.path.exists(out):
            pts = sum(1 for _ in open(out))
        best = dt if best is None else min(best, dt)
    return {"wall_s": best, "points": pts, "converged": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--testbenches", nargs="+", default=["dc", "pulse", "sine"])
    args = ap.parse_args()

    print("[exp7] compiling Verilog-A -> OSDI")
    for m in MODELS:
        compile_osdi(m)

    results = []
    for tb in args.testbenches:
        for n in args.sizes:
            row = {"testbench": tb, "n_devices": n}
            for m in MODELS:
                row[m] = run_once(m, n, tb, args.reps)
            d, s = row["dense"]["wall_s"], row["structured"]["wall_s"]
            if d and s:
                row["speedup"] = d / s
                row["pct_faster"] = 100.0 * (d - s) / d
                print(f"  {tb:<6} N={n:<4} dense={d*1000:>8.0f} ms  "
                      f"struct={s*1000:>8.0f} ms  -> {row['pct_faster']:>5.1f}% faster "
                      f"({row['speedup']:.2f}x)")
            else:
                print(f"  {tb:<6} N={n:<4} FAILED "
                      f"(dense ok={row['dense']['converged']}, "
                      f"struct ok={row['structured']['converged']})")
            results.append(row)

    outdir = RESULTS / "exp7_simscale"
    outdir.mkdir(parents=True, exist_ok=True)
    out = {"circuit": "N memristor instances in parallel, each via a 100 ohm series R",
           "models": {"dense": "[1,16,16,1], 288 weight terms",
                      "structured": "[1,4,4,1], 24 weight terms"},
           "wall_time": f"minimum of {args.reps} runs",
           "rows": results}
    (outdir / "simulator_scaling.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[exp7] wrote {outdir / 'simulator_scaling.json'}")


if __name__ == "__main__":
    main()
