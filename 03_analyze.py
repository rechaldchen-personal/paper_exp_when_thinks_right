#!/usr/bin/env python
"""
03_analyze.py — Analyze lens traces and test hypotheses H1–H3.

WHAT IT DOES:
    1. Loads traces from GPU phase (02_run_experiment.py output)
    2. Computes per-stimulus commitment metrics:
       - ℓ_H (confidence-collapse layer): where entropy exceeds threshold
       - ℓ* (commitment layer): where target locks in as top-1
       - gap (ℓ* − ℓ_H): dissociation window (confidence without correctness)
       - oscillation: top-1 token identity changes (backtracking signature)
    3. Runs paired Wilcoxon signed-rank tests (H1, H2, H3)
    4. Generates heatmaps and summary statistics
    5. Outputs JSON report with all test results

INPUTS:
    - traces.json from 02_run_experiment.py (163 stimuli × layers × positions)
    - Pre-registration from experiments/PRE_REGISTRATION.md (locked analysis)

OUTPUTS: out/analysis_real/
    - report.json: test statistics, p-values, effect sizes
    - per_stimulus_metrics.json: ℓ_H, ℓ*, gap, oscillation per item
    - heatmaps: PNGs of entropy/rank progression

HYPOTHESES TESTED:
    H1: ℓ* later (higher) under false-lead vs straightforward (p<0.05 directional)
    H2: Oscillation higher under false-lead (p<0.05 directional)
    H3: Gap (ℓ* − ℓ_H) larger under false-lead (p<0.05 directional)

CRITICAL: Dev/Holdout Split
    - theta is computed on 60% of pairs (dev set only)
    - all metrics tested on remaining 40% (holdout set)
    - prevents p-hacking by locking analysis before seeing test data

RUNS ON: CPU only (no GPU needed)

USAGE:
    # Standard: 60/40 dev/holdout split (prevents p-hacking)
    python 03_analyze.py \
        --traces out/traces.json \
        --dev-split 0.6 \
        --outdir out/analysis_real

    # Quick validation (for testing): no split (theta on full data)
    python 03_analyze.py \
        --traces out/traces.json \
        --outdir out/analysis_test

PARAMETERS:
    --traces FILE              : input traces.json from 02_run_experiment.py
    --dev-split FRAC           : fraction for dev set (default: None = no split)
    --split-seed N             : random seed for the split (default 42)
    --band LO HI               : workspace band as fractions (default 0.25 0.90)
    --outdir DIR               : output directory (default out/analysis)
    --theta-pct PCT            : dH percentile for theta (default 80)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def band_indices(layers, lo_frac, hi_frac):
    lmin, lmax = min(layers), max(layers)
    span = max(lmax - lmin, 1)
    return [i for i, L in enumerate(layers)
            if lo_frac <= (L - lmin) / span <= hi_frac]


def metrics_for(rec, theta, lo=0.25, hi=0.90):
    layers = rec["layers"]
    band = band_indices(layers, lo, hi)
    q = rec["query_position"] % rec["n_positions"]

    dH = np.array([rec["baseline_entropy"][i] - rec["entropy"][i][q]
                   for i in range(len(layers))])
    t_rank = np.array([rec["target_rank"][i][q] for i in range(len(layers))])
    d_rank = np.array([rec["distractor_rank"][i][q] for i in range(len(layers))])
    top1 = [rec["top1_id"][i][q] for i in range(len(layers))]

    # confidence-collapse layer
    l_H = next((layers[i] for i in band if dH[i] > theta), None)

    # stable commitment layer: target top-1 from here through end of band
    l_star = None
    for j, i in enumerate(band):
        if all(t_rank[k] == 0 for k in band[j:]):
            l_star = layers[i]
            break

    # oscillation: top-1 identity changes after confidence collapse
    osc = 0
    if l_H is not None:
        after = [i for i in band if layers[i] >= l_H]
        for a, b in zip(after, after[1:]):
            if top1[a] != top1[b]:
                osc += 1

    return {
        "l_H": l_H,
        "l_star": l_star,
        "gap": (l_star - l_H) if (l_star is not None and l_H is not None) else None,
        "oscillation": osc,
        "distractor_lead_layers": int(sum(d_rank[i] < t_rank[i] for i in band)),
        "dH_query": dH.tolist(),
        "t_rank_query": t_rank.tolist(),
        "d_rank_query": d_rank.tolist(),
        "band_layers": [layers[i] for i in band],
    }


def _signed_ranks(diffs):
    """Signed ranks of non-zero differences, mid-ranks for ties (standard Wilcoxon)."""
    nz = [d for d in diffs if d != 0]
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        mid = (i + j) / 2.0 + 1.0          # mid-rank over the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    return nz, ranks


def exact_signed_rank_test(diffs, alternative="greater"):
    """Wilcoxon signed-rank test, computed exactly by enumerating sign flips.

    Implemented here rather than via scipy.stats.wilcoxon because scipy's
    `method="auto"` heuristic changed across versions: with zeros/ties at small
    n it silently falls back to a normal approximation it simultaneously warns
    is invalid, so the same data produced p=0.0625 on the GPU box (scipy from
    the run-1 image) and p=0.0339 locally (scipy 1.13.1). This routine is
    deterministic and version-independent.

    Convention: zeros are dropped (standard `zero_method="wilcox"`), ties get
    mid-ranks, and the statistic is W+ (sum of positive signed ranks). The
    exact null enumerates all 2^n sign assignments, which is the correct
    reference distribution for a paired design under the sharp null.

    `alternative="greater"` tests the PRE-REGISTERED directional prediction
    (false_lead > straightforward); see experiments/PRE_REGISTRATION.md, which
    specifies "p < 0.05 (one-tailed)" for H1-H3.
    """
    nz, ranks = _signed_ranks(diffs)
    n = len(nz)
    if n == 0:
        return {"n_nonzero": 0, "W_plus": 0.0, "p": 1.0,
                "method": "degenerate (all differences zero)"}

    w_plus = sum(r for d, r in zip(nz, ranks) if d > 0)
    total = sum(ranks)

    # Exact null via subset-sum DP over the signed ranks. Mid-ranks are
    # half-integers, so work in doubled units to stay on exact integer counts;
    # this is exact for any n we will realistically see (cost is O(n * sum)),
    # unlike 2^n enumeration.
    scaled = [int(round(2 * r)) for r in ranks]
    counts = [0] * (sum(scaled) + 1)
    counts[0] = 1
    for s in scaled:
        for v in range(len(counts) - 1, s - 1, -1):
            if counts[v - s]:
                counts[v] += counts[v - s]
    denom = 2 ** n
    w2 = int(round(2 * w_plus))
    p_greater = sum(counts[w2:]) / denom
    p_less = sum(counts[:w2 + 1]) / denom
    p_two = min(1.0, 2 * min(p_greater, p_less))
    method = f"exact (subset-sum DP, n={n})"

    p = {"greater": p_greater, "less": p_less, "two-sided": p_two}[alternative]
    return {"n_nonzero": n, "W_plus": float(w_plus),
            "W_min": float(min(w_plus, total - w_plus)),
            "p": float(p), "p_one_sided_greater": float(p_greater),
            "p_two_sided": float(p_two), "method": method}


def paired_tests(by_pair, key, alternative="greater"):
    """Paired signed-rank test on (false_lead - straightforward) per pair."""
    diffs = []
    for pid, conds in sorted(by_pair.items()):
        if "straightforward" in conds and "false_lead" in conds:
            a, b = conds["straightforward"].get(key), conds["false_lead"].get(key)
            if a is not None and b is not None:
                diffs.append(b - a)
    if len(diffs) < 5:
        return {"n_pairs": len(diffs), "note": "need >=5 complete pairs",
                "diffs": diffs}
    res = exact_signed_rank_test(diffs, alternative=alternative)
    return {"n_pairs": len(diffs), "median_diff": float(np.median(diffs)),
            "alternative": alternative, **res}


def heatmap(rec, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, n_pos = rec["layers"], rec["n_positions"]
    dH = np.array([[rec["baseline_entropy"][i] - rec["entropy"][i][t]
                    for t in range(n_pos)] for i in range(len(layers))])
    tr = np.array(rec["target_rank"]).astype(float)
    dr = np.array(rec["distractor_rank"]).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, mat, title, kw in [
        (axes[0], dH, "excess entropy ΔH (nats)", dict(cmap="viridis")),
        (axes[1], np.log10(tr + 1), "log10(target rank+1)",
         dict(cmap="magma_r")),
        (axes[2], np.log10(dr + 1), "log10(distractor rank+1)",
         dict(cmap="magma_r")),
    ]:
        im = ax.imshow(mat, aspect="auto", origin="lower", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("position")
        ax.set_ylabel("layer idx")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"{rec['pair_id']} / {rec['condition']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="out/traces.json")
    ap.add_argument("--outdir", default="out/analysis")
    ap.add_argument("--band", nargs=2, type=float, default=[0.25, 0.90],
                    metavar=("LO", "HI"))
    ap.add_argument("--theta-pct", type=float, default=80.0,
                    help="percentile of straightforward-condition dH for theta")
    ap.add_argument("--dev-split", type=float, default=None,
                    help="dev/holdout split fraction (0-1); if set, theta is computed on dev only")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="random seed for dev/holdout split")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "heatmaps").mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.traces).read_text())
    lo, hi = args.band

    # ---- optional dev/holdout split -----------------------------------------
    import random
    if args.dev_split is not None:
        random.seed(args.split_seed)
        # sorted(), not list(set(...)): Python randomizes str hashing per
        # process (PYTHONHASHSEED), so set iteration order — and therefore the
        # shuffled split — varied between runs despite the fixed seed. Run 1's
        # reported split was not reproducible; sorting makes it so.
        pair_ids = sorted(set(r["pair_id"] for r in records))
        random.shuffle(pair_ids)
        dev_cutoff = int(len(pair_ids) * args.dev_split)
        dev_ids = set(pair_ids[:dev_cutoff])
        dev_records = [r for r in records if r["pair_id"] in dev_ids]
        holdout_records = [r for r in records if r["pair_id"] not in dev_ids]
        print(f"Dev/holdout split: {len(dev_ids)} dev pairs, "
              f"{len(pair_ids) - len(dev_ids)} holdout pairs")
    else:
        dev_records = records
        holdout_records = []
        print("No dev/holdout split; theta set on full data (for exploration only)")

    # ---- theta from straightforward condition (using dev set) ---------------
    pool = []
    for r in dev_records:
        if r["condition"] != "straightforward":
            continue
        band = band_indices(r["layers"], lo, hi)
        q = r["query_position"] % r["n_positions"]
        pool += [r["baseline_entropy"][i] - r["entropy"][i][q] for i in band]
    theta = float(np.percentile(pool, args.theta_pct)) if pool else 0.0
    print(f"theta (dH threshold) = {theta:.3f} nats "
          f"({args.theta_pct:.0f}th pct of straightforward condition)")

    # ---- per-stimulus metrics (on holdout set if split was used) -----------
    records_for_analysis = holdout_records if holdout_records else dev_records
    by_pair = defaultdict(dict)
    all_metrics = []
    for r in records_for_analysis:
        m = metrics_for(r, theta, lo, hi)
        m.update({k: r[k] for k in ("pair_id", "condition", "family")})
        m["behavioral_correct"] = r["behavioral"]["target_is_top1"]
        by_pair[r["pair_id"]][r["condition"]] = m
        all_metrics.append(m)
        heatmap(r, outdir / "heatmaps" / f"{r['pair_id']}_{r['condition']}.png")

    # ---- condition summaries + paired tests --------------------------------
    def summarize(cond, key):
        vals = [m[key] for m in all_metrics
                if m["condition"] == cond and m.get(key) is not None]
        return (float(np.median(vals)), len(vals)) if vals else (None, 0)

    report = {"theta": theta, "band": [lo, hi], "conditions": {}, "tests": {}}
    for cond in ("straightforward", "false_lead", "hard_control"):
        report["conditions"][cond] = {
            k: summarize(cond, k)
            for k in ("l_star", "l_H", "gap", "oscillation",
                      "distractor_lead_layers")
        }
    for key, hyp in [("l_star", "H1: later commitment under false lead"),
                     ("gap", "H3: larger dissociation gap under false lead"),
                     ("oscillation", "H2: more oscillation under false lead")]:
        report["tests"][key] = {"hypothesis": hyp, **paired_tests(by_pair, key)}

    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    (outdir / "per_stimulus_metrics.json").write_text(
        json.dumps(all_metrics, indent=2))

    print("\n=== condition medians (value, n) ===")
    for cond, d in report["conditions"].items():
        print(f"  {cond:16s} " + "  ".join(f"{k}={v}" for k, v in d.items()))
    print("\n=== paired tests (false_lead − straightforward) ===")
    for k, t in report["tests"].items():
        print(f"  {k:12s} {t}")
    print(f"\nHeatmaps + report -> {outdir}/")


if __name__ == "__main__":
    main()
