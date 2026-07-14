"""
exp2_epsilon_sensitivity.py -- sensitivity analysis for the clustering threshold epsilon.

Answers Reviewer 1: "Justify the appropriate threshold for clustering (eps = 0.1) by
applying either theoretical reasoning or a sensitivity analysis."

We do both.

Theory. Eq. (10)-(11) cluster the *absolute magnitudes* of trained parameters under
average linkage and stop when the minimum inter-cluster distance exceeds epsilon.
epsilon therefore carries the units of |theta|, so a bare 0.1 is only meaningful
relative to the spread of the trained weights. We report the normalized threshold

    eps_hat = epsilon / std(|theta|)

which is scale-free, and we give a selection rule that does not depend on a magic
number: take the largest epsilon whose reconstruction stays within a tolerance of the
unclustered model's reference error (default 1 percentage point of rel-L2). That rule
picks epsilon automatically; the sweep below then shows where 0.1 sits relative to it.

Sensitivity. We sweep epsilon over a log grid and report, per layer as in Eq. (10),
the number of clusters, the compression ratio (Eq. 14), the relative L2 error, and the
antisymmetry residual -- so a reader can see the accuracy/footprint knee directly.

Usage:
    python exp2_epsilon_sensitivity.py --seeds 5
"""

import argparse
import copy
import json

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

from common import (BurgersObjective, RESULTS, Scorer, dense_pinn, device_of,
                    n_params, psinn, summarize, train_pinn)

EPS_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
TOLERANCE_PP = 1.0        # accuracy budget for the selection rule (percentage points of rel-L2)


def cluster_layerwise(model, eps):
    """Per-layer HAC on |theta| with an average-linkage distance cutoff at eps.

    Implements Eqs. (10)-(13): each parameter is replaced by its cluster centroid with
    the sign preserved. Returns (reconstructed model, total clusters, total params).
    """
    m = copy.deepcopy(model)
    total_clusters = 0
    total_params = 0
    with torch.no_grad():
        for p in m.parameters():
            v = p.detach().cpu().numpy().ravel()
            total_params += v.size
            if v.size == 1:
                total_clusters += 1
                continue
            a = np.abs(v).reshape(-1, 1)
            labels = fcluster(linkage(a, method="average"), t=eps, criterion="distance")
            centroids = {k: np.abs(v)[labels == k].mean() for k in np.unique(labels)}
            recon = np.array([np.sign(v[i]) * centroids[labels[i]] for i in range(v.size)],
                             dtype=np.float32)
            p.copy_(torch.tensor(recon, device=p.device).view_as(p))
            total_clusters += len(centroids)
    return m, total_clusters, total_params


def weight_scale(model):
    """std(|theta|) over all parameters -- the natural scale for epsilon."""
    v = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    return float(np.std(np.abs(v)))


def sweep(model, scorer, label):
    base = scorer.score(model)
    scale = weight_scale(model)
    rows = []
    for eps in EPS_GRID:
        m, k, tot = cluster_layerwise(model, eps)
        s = scorer.score(m)
        rows.append({
            "eps": eps,
            "eps_hat": eps / scale,                      # scale-free threshold
            "clusters": k,
            "params": tot,
            "compression_ratio": k / tot,                # Eq. (14)
            "rel_l2_pct": s["rel_l2_pct"],
            "antisymmetry": s["antisymmetry"],
            "rel_l2_delta_pp": s["rel_l2_pct"] - base["rel_l2_pct"],
        })
        print(f"    [{label}] eps={eps:<6} eps_hat={eps/scale:5.2f}  K={k:<5} "
              f"rho={k/tot:7.4f}  relL2={s['rel_l2_pct']:6.1f}%  "
              f"(+{s['rel_l2_pct']-base['rel_l2_pct']:5.1f} pp)  antisym={s['antisymmetry']:.3f}")

    # Selection rule: largest eps whose accuracy loss stays within tolerance.
    ok = [r for r in rows if r["rel_l2_delta_pp"] <= TOLERANCE_PP]
    selected = max(ok, key=lambda r: r["eps"]) if ok else None
    return {"unclustered": {**base, "params": n_params(model), "weight_std": scale},
            "sweep": rows,
            "selected_eps": selected["eps"] if selected else None,
            "selected_eps_hat": selected["eps_hat"] if selected else None,
            "selected_clusters": selected["clusters"] if selected else None}


def sweep_memristor(seeds, device):
    """The 99.6%-compression claim (13 clusters / 3482 params) is quoted at eps = 0.1.

    That number is inherited from the prior Psi-HDL work, so we re-derive it here on the
    memristor model this paper actually trains, and report what eps = 0.1 costs in test
    MSE relative to the unclustered network.
    """
    from pdes import PROBLEM_SPECS, FULL_BUDGET, BASE_REG, SEEDS, train_steps

    spec = PROBLEM_SPECS["memristor"]
    runs = []
    for seed in SEEDS[:seeds]:
        torch.manual_seed(seed); np.random.seed(seed)
        data = spec["data_fn"](seed, device)
        model, _ = train_steps(spec["model_fn"]().to(device), spec["phys_loss"], data,
                               FULL_BUDGET, BASE_REG, spec["lr"])
        base_mse = spec["test_mse"](model, seed, device)
        rows = []
        for eps in EPS_GRID:
            m, k, tot = cluster_layerwise(model, eps)
            mse = spec["test_mse"](m, seed, device)
            rows.append({"eps": eps, "clusters": k, "params": tot,
                         "compression_ratio": k / tot, "test_mse": mse,
                         "mse_ratio_vs_unclustered": mse / base_mse if base_mse > 0 else None})
            print(f"    [memristor] eps={eps:<6} K={k:<5} rho={k/tot:7.4f}  "
                  f"MSE={mse:.3e}  ({mse/base_mse:7.1f}x unclustered)")
        runs.append({"base_test_mse": base_mse, "sweep": rows})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--memristor-only", action="store_true",
                    help="only re-derive the 99.6%% memristor compression claim")
    args = ap.parse_args()

    if args.memristor_only:
        device = device_of(args.device)
        outdir = RESULTS / "exp2_epsilon"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[exp2] memristor eps sweep  device={device}  seeds={args.seeds}")
        runs = sweep_memristor(args.seeds, device)
        agg = [{"eps": eps,
                "clusters": summarize([r["sweep"][i]["clusters"] for r in runs]),
                "compression_ratio": summarize([r["sweep"][i]["compression_ratio"] for r in runs]),
                "mse_ratio_vs_unclustered": summarize(
                    [r["sweep"][i]["mse_ratio_vs_unclustered"] for r in runs]),
                } for i, eps in enumerate(EPS_GRID)]
        (outdir / "epsilon_memristor.json").write_text(
            json.dumps({"grid": agg, "per_seed": runs}, indent=2), encoding="utf-8")
        print(f"\n[exp2] wrote {outdir / 'epsilon_memristor.json'}")
        return

    device = device_of(args.device)
    scorer = Scorer(device)
    outdir = RESULTS / "exp2_epsilon"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[exp2] device={device}  seeds={args.seeds}  tolerance={TOLERANCE_PP} pp")
    per_seed = {"psinn": [], "dense": []}
    for s in range(args.seeds):
        print(f"  seed {s}:")
        obj = BurgersObjective(device, seed=0)
        torch.manual_seed(s); np.random.seed(s)
        per_seed["psinn"].append(sweep(train_pinn(psinn().to(device), obj, epochs=args.epochs),
                                       scorer, "psinn"))
        torch.manual_seed(s)
        per_seed["dense"].append(sweep(train_pinn(dense_pinn().to(device), obj, epochs=args.epochs),
                                       scorer, "dense"))

    # Aggregate across seeds at each epsilon.
    agg = {}
    for model_key, runs in per_seed.items():
        agg[model_key] = {
            "unclustered_rel_l2_pct": summarize([r["unclustered"]["rel_l2_pct"] for r in runs]),
            "weight_std": summarize([r["unclustered"]["weight_std"] for r in runs]),
            "selected_eps": summarize([r["selected_eps"] for r in runs
                                       if r["selected_eps"] is not None]),
            "grid": [
                {
                    "eps": eps,
                    "eps_hat": summarize([r["sweep"][i]["eps_hat"] for r in runs]),
                    "clusters": summarize([r["sweep"][i]["clusters"] for r in runs]),
                    "compression_ratio": summarize([r["sweep"][i]["compression_ratio"] for r in runs]),
                    "rel_l2_pct": summarize([r["sweep"][i]["rel_l2_pct"] for r in runs]),
                    "antisymmetry": summarize([r["sweep"][i]["antisymmetry"] for r in runs]),
                    "rel_l2_delta_pp": summarize([r["sweep"][i]["rel_l2_delta_pp"] for r in runs]),
                }
                for i, eps in enumerate(EPS_GRID)
            ],
        }

    out = {
        "problem": "burgers",
        "method": ("per-layer hierarchical agglomerative clustering on |theta| (average linkage), "
                   "distance cutoff at eps; parameters replaced by signed centroids (Eqs. 10-13)"),
        "selection_rule": (f"largest eps whose reconstruction stays within {TOLERANCE_PP} pp of the "
                           "unclustered model's relative L2 error"),
        "eps_hat_definition": "eps / std(|theta|) -- scale-free threshold",
        "protocol": {"seeds": args.seeds, "epochs": args.epochs, "device": str(device)},
        "results": agg,
        "per_seed": per_seed,
    }
    (outdir / "epsilon_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[exp2] wrote {outdir / 'epsilon_sensitivity.json'}")
    for k in ("psinn", "dense"):
        sel = agg[k]["selected_eps"]
        print(f"  {k}: rule selects eps = {sel['mean']:.3f} +/- {sel['std']:.3f} "
              f"(std|theta| = {agg[k]['weight_std']['mean']:.3f})")


if __name__ == "__main__":
    main()
