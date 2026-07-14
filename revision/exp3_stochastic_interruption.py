"""
exp3_stochastic_interruption.py -- interruption effects that are NOT zero by construction.

Answers Reviewer 1 ("the evaluation of interruption effects is not based on experimental
evidence, given that B->E = 0 in construction ... stochastic interruptions or imperfect
checkpointing can be used to make the practical context more relevant") and Reviewer 2
("the paper is not really measuring interruption effects, only training-budget
sensitivity ... this should be reframed").

Why the submitted B->E is zero
------------------------------
With a deterministic duty cycle, lossless optimizer-state checkpointing, and no lost
work, the interrupted run replays exactly the same update sequence as continuous
training at the reduced budget. The schedule is then inert *by construction*. Arms 2
and 3 below reproduce that identity on purpose -- and show it also holds for a
*stochastic* schedule, which localizes the cause: it is the losslessness, not the
determinism, that makes the schedule inert.

What actually breaks the identity
---------------------------------
Three mechanisms, each isolated in its own arm, all at a matched *energy* budget of
1500 executed steps (cell B of the submitted decomposition):

  rollback     power fails between checkpoints, so the work since the last checkpoint
               is lost. Energy is spent but progress is not committed. Checkpoint
               interval N in {50, 200, 1000} steps.
  optreset     the checkpoint stores parameters only; Adam moments and the step counter
               are lost on every resume (a realistic NVRAM-constrained checkpoint).
  int8ckpt     the checkpoint quantizes parameters to INT8 on save (limited NVM).
  realistic    rollback + optreset together -- the practically relevant case.

Reported as a paired percentage change in test MSE against continuous training at the
same budget (cell B), with 95% bootstrap CIs over the paper's 10 seeds.

Usage:
    python exp3_stochastic_interruption.py --seeds 10
    python exp3_stochastic_interruption.py --problems burgers advection --seeds 10
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from common import RESULTS, device_of
from pdes import BASE_REG, HALF_BUDGET, PROBLEM_SPECS, SEEDS

DEFAULT_PROBLEMS = ["burgers", "heat", "advection", "allen_cahn"]
REG_MULT = 3                     # matched 3-omega regularization (cells B/C/E)
BLOCK = 100                      # mean on/off block length, in steps
DUTY = 0.5                       # expected duty cycle


# --------------------------------------------------------------------------
# power availability processes
# --------------------------------------------------------------------------
def power_trace(kind, n_steps, rng):
    """Boolean power availability per wall-clock slot, at ~50% duty."""
    if kind == "deterministic":
        return np.array([(i // BLOCK) % 2 == 0 for i in range(n_steps)])
    # Two-state Markov chain, mean block length BLOCK, stationary duty = DUTY.
    p_switch = 1.0 / BLOCK
    on = True
    trace = np.empty(n_steps, dtype=bool)
    for i in range(n_steps):
        trace[i] = on
        if rng.random() < p_switch:
            on = not on
    return trace


# --------------------------------------------------------------------------
# checkpoint fidelity
# --------------------------------------------------------------------------
def save_ckpt(model, opt, fidelity):
    params = copy.deepcopy(model.state_dict())
    if fidelity == "int8":
        for k, v in params.items():
            if v.dtype.is_floating_point and v.numel() > 1:
                scale = v.abs().max() / 127.0
                if scale > 0:
                    params[k] = torch.round(v / scale).clamp(-127, 127) * scale
    opt_state = None if fidelity == "optreset" else copy.deepcopy(opt.state_dict())
    return {"params": params, "opt": opt_state}


def load_ckpt(model, opt, ckpt, lr):
    model.load_state_dict(ckpt["params"])
    if ckpt["opt"] is not None:
        opt.load_state_dict(ckpt["opt"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)   # Adam moments lost
    return opt


# --------------------------------------------------------------------------
# the interrupted training loop
# --------------------------------------------------------------------------
def train_interrupted(spec, data, device, seed, energy_budget, schedule,
                      fidelity="lossless", rollback=False, ckpt_every=200):
    """Train under an interrupted power supply at a fixed *energy* budget.

    energy_budget counts steps actually executed (i.e. energy actually spent).
    Steps rolled back on a power failure still cost energy -- they just do not
    contribute progress. This is what makes B->E non-zero.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = spec["model_fn"]().to(device)
    lr = spec["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    reg = REG_MULT * BASE_REG
    phys_loss = spec["phys_loss"]

    trace = power_trace(schedule, energy_budget * 8, rng)   # generous wall-clock horizon
    ckpt = save_ckpt(model, opt, fidelity)
    executed = 0
    committed = 0
    since_ckpt = 0
    outages = 0
    lost = 0
    prev_on = True

    for on in trace:
        if executed >= energy_budget:
            break
        if on:
            if not prev_on:                       # resume from checkpoint
                opt = load_ckpt(model, opt, ckpt, lr)
            opt.zero_grad()
            loss = phys_loss(model, data) + reg * sum(p.pow(2).sum() for p in model.parameters())
            loss.backward()
            opt.step()
            executed += 1
            since_ckpt += 1
            committed += 1
            if ckpt_every and since_ckpt >= ckpt_every:
                ckpt = save_ckpt(model, opt, fidelity)
                since_ckpt = 0
        else:
            if prev_on:                           # power just failed
                outages += 1
                if rollback and since_ckpt > 0:
                    load_ckpt(model, opt, ckpt, lr)   # uncommitted work is lost
                    lost += since_ckpt
                    committed -= since_ckpt
                    since_ckpt = 0
                else:
                    ckpt = save_ckpt(model, opt, fidelity)   # checkpoint at failure
                    since_ckpt = 0
        prev_on = on

    return model, {"executed": executed, "committed": committed,
                   "lost_steps": lost, "outages": outages}


def train_continuous(spec, data, device, seed, n_steps):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = spec["model_fn"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    reg = REG_MULT * BASE_REG
    for _ in range(n_steps):
        opt.zero_grad()
        loss = spec["phys_loss"](model, data) + reg * sum(p.pow(2).sum() for p in model.parameters())
        loss.backward()
        opt.step()
    return model


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
ARMS = [
    # (name, schedule, fidelity, rollback, ckpt_every)
    # The checkpoint interval is swept across the mean uptime (BLOCK = 100 steps):
    # 10 and 50 are below it, 200 and 1000 above it. Above it the system can lose
    # every uncommitted segment and stall -- the intermittent-computing livelock.
    ("det_lossless",        "deterministic", "lossless", False, 200),
    ("stoch_lossless",      "stochastic",    "lossless", False, 200),
    ("stoch_rollback_10",   "stochastic",    "lossless", True,  10),
    ("stoch_rollback_50",   "stochastic",    "lossless", True,  50),
    ("stoch_rollback_200",  "stochastic",    "lossless", True,  200),
    ("stoch_rollback_1000", "stochastic",    "lossless", True,  1000),
    ("stoch_optreset",      "stochastic",    "optreset", False, 200),
    ("stoch_int8ckpt",      "stochastic",    "int8",     False, 200),
    ("stoch_realistic",     "stochastic",    "optreset", True,  200),
]


def paired_pct(arm_mse, base_mse, n_boot=10_000):
    """Mean per-seed % change vs the matched-budget continuous baseline, with CI."""
    ratios = 100.0 * (np.asarray(arm_mse) - np.asarray(base_mse)) / np.asarray(base_mse)
    rng = np.random.default_rng(42)
    boots = rng.choice(ratios, size=(n_boot, len(ratios)), replace=True).mean(1)
    return {"pct_change": float(ratios.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "resolved": bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0),
            "per_seed": [float(r) for r in ratios]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=HALF_BUDGET)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = device_of(args.device)
    seeds = SEEDS[:args.seeds]
    outdir = RESULTS / "exp3_interruption"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[exp3] device={device}  problems={args.problems}  seeds={len(seeds)}  "
          f"energy budget={args.budget} executed steps")
    out = {"protocol": {
        "energy_budget_steps": args.budget, "reg_mult": REG_MULT, "duty": DUTY,
        "mean_block_steps": BLOCK, "seeds": seeds, "device": str(device),
        "baseline": "continuous training at the same budget (cell B)",
        "note": ("rolled-back steps still consume energy; the energy budget counts executed "
                 "steps, so rollback arms commit fewer than `budget` steps"),
    }, "problems": {}}

    t0 = time.time()
    for prob in args.problems:
        spec = PROBLEM_SPECS[prob]
        print(f"\n  {prob}:")
        base, arms = [], {a[0]: [] for a in ARMS}
        stats = {a[0]: [] for a in ARMS}

        for seed in seeds:
            data = spec["data_fn"](seed, device)
            m = train_continuous(spec, data, device, seed, args.budget)
            base.append(spec["test_mse"](m, seed, device))

            for name, sched, fid, rb, ck in ARMS:
                m, st = train_interrupted(spec, data, device, seed, args.budget,
                                          sched, fid, rb, ck)
                arms[name].append(spec["test_mse"](m, seed, device))
                stats[name].append(st)

        res = {"continuous_B_mse": {"mean": float(np.mean(base)),
                                    "per_seed": [float(b) for b in base]}}
        for name, *_ in ARMS:
            r = paired_pct(arms[name], base)
            r["committed_steps_mean"] = float(np.mean([s["committed"] for s in stats[name]]))
            r["lost_steps_mean"] = float(np.mean([s["lost_steps"] for s in stats[name]]))
            r["outages_mean"] = float(np.mean([s["outages"] for s in stats[name]]))
            res[name] = r
            flag = "resolved" if r["resolved"] else "n.s."
            print(f"    {name:<22} {r['pct_change']:+9.1f}%  "
                  f"[{r['ci95'][0]:+8.1f},{r['ci95'][1]:+8.1f}]  {flag:<9} "
                  f"committed={r['committed_steps_mean']:.0f}/{args.budget}")
        out["problems"][prob] = res

    out["elapsed_s"] = round(time.time() - t0, 1)
    (outdir / "stochastic_interruption.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[exp3] wrote {outdir / 'stochastic_interruption.json'}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
