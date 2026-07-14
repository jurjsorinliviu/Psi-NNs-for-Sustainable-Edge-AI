"""
exp4_distill_clustering.py -- does the compressed model actually work?

Re-derives the compression claim on the two models whose cluster counts the paper
quotes (Burgers and the memristor), comparing three pipelines at each threshold eps:

  unclustered   the trained network (reference accuracy)
  posthoc       cluster and stop -- what the submitted paper does
  distilled     distil -> cluster -> retrain the K centroids (Ref. [23]'s pipeline)

The question this answers is the one the hardware extraction depends on: at how many
clusters K does the compressed model still *work*? The memory footprint of Sec. 3.2.3
is sum_l K_l * S_param, so it is only meaningful at a K where accuracy survives.

Usage:
    python exp4_distill_clustering.py --problem burgers --seeds 5
    python exp4_distill_clustering.py --problem memristor --seeds 5
"""

import argparse
import json
import time

import numpy as np
import torch

from common import (BurgersObjective, RESULTS, Scorer, dense_pinn, device_of,
                    n_params, psinn, summarize, train_pinn)
from distill_cluster import (ClusteredModel, cluster_posthoc, distill_student,
                             retrain_centroids)

EPS_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
BYTES_PER_PARAM = 4


# --------------------------------------------------------------------------
def setup_burgers(seed, device, epochs, lr):
    obj = BurgersObjective(device, seed=0)
    scorer = Scorer(device)

    torch.manual_seed(seed); np.random.seed(seed)
    teacher = train_pinn(dense_pinn().to(device), obj, epochs=epochs, lr=lr)

    torch.manual_seed(seed)
    student = psinn().to(device)

    obj_fn = lambda m: obj(m)
    inputs = (obj.X.detach(),)

    def score(m):
        s = scorer.score(m)
        return {"rel_l2_pct": s["rel_l2_pct"], "antisymmetry": s["antisymmetry"]}

    # "better" = lower relative L2
    return dict(teacher=teacher, student=student, obj_fn=obj_fn, inputs=inputs,
                score=score, metric="rel_l2_pct", lr=lr)


def setup_memristor(seed, device, epochs, lr):
    from pdes import BASE_REG, PROBLEM_SPECS, SEEDS, train_steps
    spec = PROBLEM_SPECS["memristor"]
    s = SEEDS[seed % len(SEEDS)]
    data = spec["data_fn"](s, device)

    torch.manual_seed(s); np.random.seed(s)
    teacher, _ = train_steps(spec["model_fn"]().to(device), spec["phys_loss"], data,
                             epochs, BASE_REG, spec["lr"])
    torch.manual_seed(s)
    student = spec["model_fn"]().to(device)

    obj_fn = lambda m: spec["phys_loss"](m, data)
    inputs = (data["V"], data["x"])

    def score(m):
        return {"test_mse": spec["test_mse"](m, s, device)}

    return dict(teacher=teacher, student=student, obj_fn=obj_fn, inputs=inputs,
                score=score, metric="test_mse", lr=spec["lr"])


SETUPS = {"burgers": setup_burgers, "memristor": setup_memristor}


# --------------------------------------------------------------------------
def run_seed(problem, seed, device, epochs, lr):
    cfg = SETUPS[problem](seed, device, epochs, lr)
    obj_fn, inputs, score, lr_ = cfg["obj_fn"], cfg["inputs"], cfg["score"], cfg["lr"]

    # --- reference: the plainly trained student (no distillation, no clustering) ---
    torch.manual_seed(seed)
    plain = cfg["student"]
    opt = torch.optim.Adam(plain.parameters(), lr=lr_)
    for _ in range(epochs):
        opt.zero_grad(); obj_fn(plain).backward(); opt.step()
    ref = score(plain)
    ref["params"] = n_params(plain)
    ref["bytes"] = ref["params"] * BYTES_PER_PARAM

    # --- stage 2: distilled student (weights driven toward discrete levels) ---
    torch.manual_seed(seed)
    student2 = SETUPS[problem](seed, device, 1, lr)["student"]      # fresh init
    distilled = distill_student(student2, cfg["teacher"], obj_fn, inputs,
                                epochs=epochs, lr=lr_, alpha=1.0, reg=1e-3)
    dist_ref = score(distilled)

    rows = []
    for eps in EPS_GRID:
        # (a) what the paper does: cluster the plainly trained net and stop
        ph, k_ph = cluster_posthoc(plain, eps)
        s_ph = score(ph)

        # (b) cluster the plainly trained net, then RETRAIN THE CENTROIDS.
        #     This isolates the retraining stage from distillation.
        cr = ClusteredModel(plain, eps).to(device)
        k_cr = cr.n_clusters()
        s_cr_before = score(cr)
        retrain_centroids(cr, obj_fn, epochs=epochs, lr=lr_)
        s_cr = score(cr)

        # (c) Ref. [23] in full: distil -> cluster -> retrain the centroids
        cm = ClusteredModel(distilled, eps).to(device)
        k_dc = cm.n_clusters()
        retrain_centroids(cm, obj_fn, epochs=epochs, lr=lr_)
        s_dc = score(cm)

        rows.append({
            "eps": eps,
            "posthoc": {"clusters": k_ph, **s_ph},
            "clustered_retrained": {"clusters": k_cr, "before_retrain": s_cr_before, **s_cr},
            "distilled_clustered": {"clusters": k_dc, **s_dc},
        })
        m = cfg["metric"]
        print(f"    eps={eps:<5} | posthoc K={k_ph:<4} {s_ph[m]:.3e}"
              f" | +retrain K={k_cr:<4} {s_cr[m]:.3e}"
              f" | distil+retrain K={k_dc:<4} {s_dc[m]:.3e}")

    return {"unclustered": ref, "distilled_unclustered": dist_ref, "sweep": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(SETUPS), default="burgers")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = device_of(args.device)
    outdir = RESULTS / "exp4_distill"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[exp4] {args.problem}  device={device}  seeds={args.seeds}")

    t0 = time.time()
    runs = []
    for s in range(args.seeds):
        print(f"  seed {s}:")
        runs.append(run_seed(args.problem, s, device, args.epochs, args.lr))

    metric = "rel_l2_pct" if args.problem == "burgers" else "test_mse"
    agg = {
        "unclustered": {
            "params": runs[0]["unclustered"]["params"],
            "bytes": runs[0]["unclustered"]["bytes"],
            metric: summarize([r["unclustered"][metric] for r in runs]),
        },
        "grid": [{
            "eps": eps,
            **{arm: {
                "clusters": summarize([r["sweep"][i][arm]["clusters"] for r in runs]),
                # true device weight memory: K centroids (4 B each) PLUS one index byte
                # per parameter for the relation matrix R, which Eq. (23) omits
                "weight_bytes_true": summarize(
                    [r["sweep"][i][arm]["clusters"] * BYTES_PER_PARAM
                     + runs[0]["unclustered"]["params"] for r in runs]),
                metric: summarize([r["sweep"][i][arm][metric] for r in runs]),
            } for arm in ("posthoc", "clustered_retrained", "distilled_clustered")},
        } for i, eps in enumerate(EPS_GRID)],
    }
    if args.problem == "burgers":
        for i, eps in enumerate(EPS_GRID):
            for arm in ("posthoc", "clustered_retrained", "distilled_clustered"):
                agg["grid"][i][arm]["antisymmetry"] = summarize(
                    [r["sweep"][i][arm]["antisymmetry"] for r in runs])

    out = {"problem": args.problem, "metric": metric,
           "pipeline": ("posthoc = cluster and stop (submitted paper); "
                        "distilled_clustered = distil -> cluster -> retrain centroids (Ref. [23])"),
           "protocol": {"seeds": args.seeds, "epochs": args.epochs, "device": str(device)},
           "summary": agg, "per_seed": runs, "elapsed_s": round(time.time() - t0, 1)}
    (outdir / f"distill_clustering_{args.problem}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n[exp4] wrote {outdir / f'distill_clustering_{args.problem}.json'}")
    ref = agg["unclustered"][metric]["mean"]
    print(f"  unclustered: {agg['unclustered']['params']} params "
          f"({agg['unclustered']['bytes']} B), {metric}={ref:.3e}")
    print(f"\n  {'eps':>6} {'K(posthoc)':>11} {'posthoc':>12} {'K(distil)':>10} "
          f"{'distilled':>12} {'bytes':>8}")
    for g in agg["grid"]:
        print(f"  {g['eps']:>6} {g['posthoc']['clusters']['mean']:>11.0f} "
              f"{g['posthoc'][metric]['mean']:>12.2e} "
              f"{g['distilled_clustered']['clusters']['mean']:>10.0f} "
              f"{g['distilled_clustered'][metric]['mean']:>12.2e} "
              f"{g['distilled_clustered']['bytes']['mean']:>8.0f}")


if __name__ == "__main__":
    main()
