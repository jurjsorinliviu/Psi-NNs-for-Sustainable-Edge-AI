"""
exp1_compression_baselines.py -- strong compression baselines on the Burgers PDE.

Answers Reviewer 1 ("add compression baselines such as iterative pruning,
fine-tuning or quantization-aware training") and Reviewer 2 ("compare against
stronger compression baselines such as pruning with retraining, quantization-aware
training, structured pruning, low-rank compression, distillation, and size-matched
compact dense networks").

Every baseline is trained on the identical forward-PINN objective and scored on the
same two axes as the submitted Table 3: relative L2 against a finite-difference
reference, and the odd-in-x antisymmetry residual the physics requires.

Compression baselines that start from the trained dense network are given a
*larger* total training budget than the structured Psi-NN (initial training plus
fine-tuning). This is deliberate: the point is to try hard to beat the structured
model, so any surviving gap is not an artifact of a starved baseline.

Footprint target: the structured Psi-NN's 1,937 parameters (7.6 KB at fp32).

Usage:
    python exp1_compression_baselines.py --seeds 5 --epochs 3000
"""

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

from common import (MLP, BurgersObjective, RESULTS, Scorer, dense_pinn, device_of,
                    n_params, psinn, summarize, train_pinn, weight_bytes)

TARGET_PARAMS = 1937          # the structured Psi-NN footprint we match against
SIZE_MATCHED_WIDTH = 24       # 4x24 dense -> 1,897 params (closest match under 1,937)
LOWRANK_RANK = 5              # -> 1,851 params
DISTILL_ALPHA = 1.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def weight_names(model):
    return [n for n, _ in model.named_parameters() if "weight" in n]


def global_magnitude_mask(model, sparsity):
    """Global magnitude mask over weight tensors (biases untouched)."""
    ws = {n: p for n, p in model.named_parameters() if "weight" in n}
    allw = torch.cat([p.detach().abs().flatten() for p in ws.values()])
    k = int(len(allw) * sparsity)
    thr = torch.kthvalue(allw, k).values if k > 0 else torch.tensor(-1.0)
    return {n: (p.detach().abs() > thr).float() for n, p in ws.items()}


def apply_mask(model, mask):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in mask:
                p.mul_(mask[n])
    return model


def nonzero_params(model):
    """Parameters that actually have to be stored (non-zero weights + all biases)."""
    tot = 0
    for n, p in model.named_parameters():
        tot += int((p != 0).sum().item()) if "weight" in n else p.numel()
    return tot


def sparsity_for_target(dense_model, target=TARGET_PARAMS):
    """Sparsity that leaves `target` stored parameters in the dense net."""
    n_w = sum(p.numel() for n, p in dense_model.named_parameters() if "weight" in n)
    n_b = sum(p.numel() for n, p in dense_model.named_parameters() if "bias" in n)
    keep_w = max(target - n_b, 1)
    return 1.0 - keep_w / n_w


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def b_iterative_prune_ft(dense, obj, scorer, sparsity, cycles=5, ft_epochs=600, lr=1e-3):
    """Gradual magnitude pruning with fine-tuning between cycles (R1 + R2)."""
    m = copy.deepcopy(dense)
    for c in range(1, cycles + 1):
        s = sparsity * c / cycles                       # linear sparsity ramp
        mask = global_magnitude_mask(m, s)
        apply_mask(m, mask)
        train_pinn(m, obj, epochs=ft_epochs, lr=lr, mask=mask)
    return m


class StructuredMLP(nn.Module):
    """Dense MLP with whole neurons removed (a genuinely smaller, dense network)."""

    def __init__(self, dense, keep):
        super().__init__()
        lins = [l for l in dense.net if isinstance(l, nn.Linear)]
        dims = [2] + [len(k) for k in keep] + [1]
        self.inner = MLP(dims)
        new_lins = [l for l in self.inner.net if isinstance(l, nn.Linear)]
        with torch.no_grad():
            prev = torch.arange(2)
            for i, l in enumerate(lins):
                out_idx = keep[i] if i < len(keep) else torch.arange(1)
                new_lins[i].weight.copy_(l.weight[out_idx][:, prev])
                new_lins[i].bias.copy_(l.bias[out_idx])
                prev = out_idx

    def forward(self, x):
        return self.inner(x)


def b_structured_prune_ft(dense, obj, scorer, width, ft_epochs=1500, lr=1e-3):
    """Neuron-level (structured) pruning by L1 importance, then fine-tune."""
    lins = [l for l in dense.net if isinstance(l, nn.Linear)]
    keep = []
    for i, l in enumerate(lins[:-1]):                   # hidden layers only
        imp = l.weight.detach().abs().sum(1) + lins[i + 1].weight.detach().abs().sum(0)
        keep.append(torch.sort(torch.topk(imp, width).indices).values)
    m = StructuredMLP(dense, keep).to(next(dense.parameters()).device)
    train_pinn(m, obj, epochs=ft_epochs, lr=lr)
    return m


class LowRankMLP(nn.Module):
    """Hidden weight matrices factorized as U V (rank r), initialized by truncated SVD."""

    def __init__(self, dense, rank):
        super().__init__()
        lins = [l for l in dense.net if isinstance(l, nn.Linear)]
        self.first = nn.Linear(lins[0].in_features, lins[0].out_features)
        self.last = nn.Linear(lins[-1].in_features, lins[-1].out_features)
        self.U, self.V, self.B = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        with torch.no_grad():
            self.first.weight.copy_(lins[0].weight); self.first.bias.copy_(lins[0].bias)
            self.last.weight.copy_(lins[-1].weight); self.last.bias.copy_(lins[-1].bias)
            for l in lins[1:-1]:
                u, s, vh = torch.linalg.svd(l.weight.detach(), full_matrices=False)
                r = min(rank, s.numel())
                self.U.append(nn.Parameter(u[:, :r] * s[:r]))
                self.V.append(nn.Parameter(vh[:r, :]))
                self.B.append(nn.Parameter(l.bias.detach().clone()))

    def forward(self, x):
        h = torch.tanh(self.first(x))
        for u, v, b in zip(self.U, self.V, self.B):
            h = torch.tanh(h @ v.T @ u.T + b)
        return self.last(h)


def b_lowrank_ft(dense, obj, rank=LOWRANK_RANK, ft_epochs=1500, lr=1e-3):
    m = LowRankMLP(dense, rank).to(next(dense.parameters()).device)
    train_pinn(m, obj, epochs=ft_epochs, lr=lr)
    return m


class FakeQuant(torch.autograd.Function):
    """Per-tensor symmetric INT8 fake-quantization with a straight-through estimator."""

    @staticmethod
    def forward(ctx, w):
        scale = w.abs().max() / 127.0
        return torch.round(w / scale).clamp(-127, 127) * scale if scale > 0 else w

    @staticmethod
    def backward(ctx, g):
        return g


class QATMLP(nn.Module):
    """Dense 4x50 trained with INT8 fake-quant in the loop (quantization-aware)."""

    def __init__(self, width=50, depth=4):
        super().__init__()
        dims = [2] + [width] * depth + [1]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        h = x
        for i, l in enumerate(self.lins):
            h = nn.functional.linear(h, FakeQuant.apply(l.weight), l.bias)
            if i < len(self.lins) - 1:
                h = torch.tanh(h)
        return h


def b_distill_compact(teacher, obj, device, width=SIZE_MATCHED_WIDTH, epochs=3000, lr=1e-3):
    """Dense teacher -> size-matched compact dense student (physics + distillation loss)."""
    student = MLP([2] + [width] * 4 + [1]).to(device)
    with torch.no_grad():
        t_pred = teacher(obj.X.detach())

    def distill(m):
        return DISTILL_ALPHA * torch.mean((m(obj.X.detach()) - t_pred) ** 2)

    train_pinn(student, obj, epochs=epochs, lr=lr, extra_loss=distill)
    return student


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_seed(seed, epochs, lr, device, scorer):
    obj = BurgersObjective(device, seed=0)     # same collocation set for every model
    rows = {}

    def record(key, model, params=None, bytes_per=4, note=""):
        p = params if params is not None else nonzero_params(model)
        s = scorer.score(model)
        rows[key] = {"params": p, "weight_bytes": weight_bytes(p, bytes_per),
                     "note": note, **s}
        print(f"    {key:<26} params={p:<6} relL2={s['rel_l2_pct']:6.1f}%  "
              f"antisym={s['antisymmetry']:.3f}")
        return model

    torch.manual_seed(seed); np.random.seed(seed)
    struct = train_pinn(psinn().to(device), obj, epochs=epochs, lr=lr)
    record("psinn_structured", struct, note="directly trained structured architecture")

    torch.manual_seed(seed)
    dense = train_pinn(dense_pinn().to(device), obj, epochs=epochs, lr=lr)
    record("dense_4x50", dense, note="submitted dense baseline")

    torch.manual_seed(seed)
    sized = train_pinn(MLP([2] + [SIZE_MATCHED_WIDTH] * 4 + [1]).to(device),
                       obj, epochs=epochs, lr=lr)
    record("dense_size_matched", sized, note=f"4x{SIZE_MATCHED_WIDTH} dense, footprint-matched")

    # Single-shot magnitude pruning at the matched footprint (the submitted method).
    s_target = sparsity_for_target(dense)
    oneshot = apply_mask(copy.deepcopy(dense), global_magnitude_mask(dense, s_target))
    record("prune_oneshot", oneshot, note=f"single-shot global magnitude, {s_target:.1%} sparsity")

    torch.manual_seed(seed)
    iterative = b_iterative_prune_ft(dense, obj, scorer, s_target)
    record("prune_iterative_ft", iterative,
           note=f"gradual magnitude pruning to {s_target:.1%} + fine-tuning (5 cycles x 600 ep)")

    torch.manual_seed(seed)
    structured = b_structured_prune_ft(dense, obj, scorer, SIZE_MATCHED_WIDTH)
    record("prune_structured_ft", structured,
           note=f"neuron-level L1 pruning to width {SIZE_MATCHED_WIDTH} + 1500 ep fine-tuning")

    torch.manual_seed(seed)
    lowrank = b_lowrank_ft(dense, obj)
    record("lowrank_svd_ft", lowrank,
           note=f"truncated SVD rank {LOWRANK_RANK} + 1500 ep fine-tuning")

    torch.manual_seed(seed)
    qat = train_pinn(QATMLP().to(device), obj, epochs=epochs, lr=lr)
    record("qat_int8", qat, params=n_params(qat), bytes_per=1,
           note="quantization-aware training, INT8 weights (1 byte/param)")

    torch.manual_seed(seed)
    distilled = b_distill_compact(dense, obj, device)
    record("distill_compact", distilled,
           note=f"dense teacher -> 4x{SIZE_MATCHED_WIDTH} student, physics + distillation loss")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = device_of(args.device)
    scorer = Scorer(device)
    outdir = RESULTS / "exp1_compression" if args.outdir is None else __import__("pathlib").Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[exp1] device={device}  seeds={args.seeds}  epochs={args.epochs}")
    per_seed = []
    t0 = time.time()
    for s in range(args.seeds):
        print(f"  seed {s}:")
        per_seed.append(run_seed(s, args.epochs, args.lr, device, scorer))

    keys = list(per_seed[0].keys())
    summary = {}
    for k in keys:
        summary[k] = {
            "params": per_seed[0][k]["params"],
            "weight_bytes": per_seed[0][k]["weight_bytes"],
            "note": per_seed[0][k]["note"],
            "rel_l2_pct": summarize([r[k]["rel_l2_pct"] for r in per_seed]),
            "antisymmetry": summarize([r[k]["antisymmetry"] for r in per_seed]),
        }

    out = {
        "problem": "burgers",
        "reference": "finite-difference numerical solution (u(x,0) = -sin(pi x))",
        "metric": ("rel_l2_pct = relative L2 vs FD (%); antisymmetry = mean|f(x)+f(-x)|/mean|f| "
                   "(0 = exactly odd in x, 2.0 = non-zero constant)"),
        "protocol": {
            "seeds": args.seeds, "epochs": args.epochs, "lr": args.lr,
            "device": str(device),
            "footprint_target_params": TARGET_PARAMS,
            "budget_note": ("compression baselines derived from the dense network receive the "
                            "initial 3000-epoch training PLUS fine-tuning, i.e. a strictly larger "
                            "budget than the structured Psi-NN"),
        },
        "baselines": summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (outdir / "compression_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n[exp1] wrote {outdir / 'compression_baselines.json'}  ({out['elapsed_s']}s)")
    print(f"\n{'baseline':<26}{'params':>8}{'KB':>7}{'rel L2 %':>20}{'antisym':>18}")
    for k, v in summary.items():
        r, a = v["rel_l2_pct"], v["antisymmetry"]
        print(f"{k:<26}{v['params']:>8}{v['weight_bytes']/1024:>7.1f}"
              f"{r['mean']:>11.1f} +/-{r['std']:<6.1f}"
              f"{a['mean']:>11.3f} +/-{a['std']:<6.3f}")


if __name__ == "__main__":
    main()
