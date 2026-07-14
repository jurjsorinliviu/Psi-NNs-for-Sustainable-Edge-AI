"""
exp6_scaling_frontier.py -- does the structured architecture ever change the platform?

Answers Reviewer 2's central objection: "the reported ~45x lifecycle carbon reduction
mainly comes from replacing a Jetson-class module with a microcontroller, not from the
physics-structured model itself. The authors even acknowledge that the dense Burgers
model already fits the MCU, so the structured representation did not actually enable the
platform change" -- and the related one: "the recommender simply chooses the lowest-power
platform ... the only case where compute becomes meaningful is a hypothetical larger
surrogate, not an actual validated experiment."

Both are fair. At the demonstrated size every model fits every platform, so the
recommender does no work. The honest test is therefore not another small model but the
FRONTIER: as the surrogate grows, which family stops fitting the target first, and does
the structured family buy a platform tier?

We scale the Burgers Psi-NN (node_num) and a dense PINN (width) together, train both,
compile both to Cortex-M4 firmware against the real nRF52840 memory map, and record the
size at which each stops fitting. Crucially the fit test is the linker's, not ours: a
model that does not fit fails to link.

This experiment can come out against the paper's thesis. If the structured family buys no
tier, we report that.

Usage:
    python exp6_scaling_frontier.py --sizes 16 32 64 96 128
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "exp5_mcu"))

from common import (MLP, BurgersObjective, RESULTS, Scorer, device_of, n_params,
                    psinn, train_pinn)                                    # noqa: E402
from exp5_mcu.export_c import export_mlp, export_psinn_burgers            # noqa: E402
from exp5_mcu.run_mcu_bench import (TARGETS, build, make_test_set, run_qemu,  # noqa: E402
                                    score_device, size_of, write_test_data,
                                    INSTR_PER_TICK)

# nRF52840: the throughput requirement of the paper's worked surrogate example
GRID = 128 * 128
RATE_HZ = 100
INFERENCES_PER_S = GRID * RATE_HZ          # 1.64e6


def dense_width_for(target_params):
    """Width w of a 4-hidden-layer dense MLP with ~target_params: 3w^2 + 7w + 1."""
    w = int((-7 + np.sqrt(49 + 12 * (target_params - 1))) / 6)
    return max(w, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt", default="O2")
    args = ap.parse_args()

    # Train on the GPU: at node_num=128 the structured net has ~1.2e5 parameters and the
    # forward-PINN objective needs second-order autograd, which is far too slow on CPU.
    dev = device_of("auto")
    scorer = Scorer(dev)
    obj = BurgersObjective(dev, seed=0)
    pts = make_test_set(128)
    cpu = device_of("cpu")
    ref_scorer = Scorer(cpu)
    ref_fn = lambda p: ref_scorer._fd_ref(torch.tensor(p)).cpu().numpy().ravel().astype(np.float64)

    workdir = RESULTS / "exp6_scaling" / "build"
    workdir.mkdir(parents=True, exist_ok=True)
    write_test_data(workdir / "test_data.h", pts)

    rows = []
    for n in args.sizes:
        # structured: node_num = n
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        m_s = train_pinn(psinn(node_num=n).to(dev), obj, epochs=args.epochs)
        p_s = n_params(m_s)

        # dense: matched parameter count, so the comparison is at equal capacity
        w = dense_width_for(p_s)
        torch.manual_seed(args.seed)
        m_d = train_pinn(MLP([2] + [w] * 4 + [1]).to(dev), obj, epochs=args.epochs)
        p_d = n_params(m_d)

        print(f"\n=== node_num={n}: structured {p_s} params | dense 4x{w} {p_d} params ===")
        entry = {"node_num": n, "structured_params": p_s, "dense_width": w,
                 "dense_params": p_d,
                 "host": {"structured": scorer.score(m_s), "dense": scorer.score(m_d)}}

        for tag, model, exporter in (
                ("structured", m_s, lambda p: export_psinn_burgers(m_s, p, "float32")),
                ("dense", m_d, lambda p: export_mlp(m_d, p, "float32", name="model"))):
            cname = "psinn" if tag == "structured" else "model"
            cfile = workdir / f"{tag}_{n}.c"
            meta = exporter(cfile)
            header = (f"#define MODEL_FORWARD {cname}_forward\n#define MODEL_NOUT 1\n"
                      f"void {cname}_forward(const float *in, float *out);\n")
            elf, kind, detail = build("nrf52840", cfile, header, workdir, args.opt)
            if elf is None and kind == "does_not_fit":
                entry[tag] = {"fits_nrf52840": False, **detail, "params": meta["params"]}
                print(f"  {tag:<11} DOES NOT FIT nRF52840: {detail['region']} "
                      f"overflowed by {detail['overflow_bytes']} B")
                continue
            if elf is None:
                entry[tag] = {"build_error": detail}
                print(f"  {tag:<11} BUILD ERROR: {detail[:150]}")
                continue

            sz = size_of(elf)
            run, err = run_qemu(elf, TARGETS["nrf52840"]["machine"], TARGETS["nrf52840"]["cpu"])
            if run is None:
                entry[tag] = {"fits_nrf52840": True, **sz, "run_error": err}
                print(f"  {tag:<11} fits (flash={sz['flash_bytes']}B) but run failed")
                continue

            ipi = run["instructions"] / len(pts)
            acc = score_device(run["outputs"], pts, ref_fn)
            # throughput feasibility at the paper's 128x128 @ 100 Hz workload
            ops_per_s = ipi * INFERENCES_PER_S
            tops = ops_per_s / 1e12
            entry[tag] = {"fits_nrf52840": True, **sz, "params": meta["params"],
                          "instructions_per_inference": ipi,
                          "required_tops_at_128x128_100Hz": tops,
                          "device_accuracy": acc}
            print(f"  {tag:<11} flash={sz['flash_bytes']:>7}B ram={sz['ram_bytes']:>6}B "
                  f"instr/inf={ipi:>9.0f} relL2={acc['rel_l2_pct']:5.1f}% "
                  f"req={tops:.4f} TOPS")
        rows.append(entry)

    out = {"target": "nRF52840 (Cortex-M4F, 1024 KB flash / 256 KB RAM)",
           "opt": args.opt, "seed": args.seed, "epochs": args.epochs,
           "throughput_workload": {"grid": "128x128", "rate_hz": RATE_HZ,
                                   "inferences_per_s": INFERENCES_PER_S,
                                   "nrf52840_tops": 0.026},
           "note": ("fit is decided by the linker against the real memory map; "
                    "required TOPS uses MEASURED instructions per inference, not the "
                    "architectural operation count"),
           "rows": rows}
    outdir = RESULTS / "exp6_scaling"
    (outdir / "scaling_frontier.json").write_text(json.dumps(out, indent=2, default=float),
                                                  encoding="utf-8")
    print(f"\n[exp6] wrote {outdir / 'scaling_frontier.json'}")


if __name__ == "__main__":
    main()
