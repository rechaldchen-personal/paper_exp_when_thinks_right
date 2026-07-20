#!/usr/bin/env python3
"""Generate publication figures from out/analysis_real/ (Phase 5)."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

report = json.load(open("out/analysis_real/report.json"))
metrics = json.load(open("out/analysis_real/per_stimulus_metrics.json"))
conditions = ["straightforward", "false_lead", "hard_control"]
out = Path("out/figures")
out.mkdir(parents=True, exist_ok=True)


def collect(field):
    data = {c: [] for c in conditions}
    for m in metrics:
        v = m.get(field)
        if v is None:
            continue
        data[m["condition"]].append(v)
    return data


def boxplot(field, ylabel, title, color, test_key, outname, unit_label):
    data = collect(field)
    plot_conds = [c for c in conditions if data[c]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [data[c] for c in plot_conds],
        tick_labels=plot_conds,
        patch_artist=True,
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, c in enumerate(plot_conds, start=1):
        ys = data[c]
        xs = np.random.normal(i, 0.04, size=len(ys))
        ax.scatter(xs, ys, alpha=0.45, s=28, color="black", zorder=3)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Condition", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    test = report["tests"][test_key]
    support = "SUPPORTED" if test["p"] < 0.05 else "NOT SIG. (p>=0.05)"
    stats_text = (
        f"exact signed-rank, one-tailed\n"
        f"p={test['p']:.4g}\n"
        f"Median Δ={test['median_diff']:.1f} {unit_label}\n"
        f"n_pairs={test['n_pairs']} (nonzero {test['n_nonzero']})\n{support}"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )
    plt.tight_layout()
    path = out / outname
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def main():
    boxplot(
        "l_star",
        "Commitment Layer ℓ*",
        "H1: Later Commitment Under False Lead",
        "#3498db",
        "l_star",
        "H1_commitment_layer.png",
        "layers",
    )
    boxplot(
        "oscillation",
        "Oscillation Depth (# top-1 changes)",
        "H2: Internal Backtracking Signature",
        "#e74c3c",
        "oscillation",
        "H2_oscillation.png",
        "changes",
    )
    boxplot(
        "gap",
        "Dissociation Gap (ℓ* − ℓ_H)",
        "H3: Confidently-Wrong Window",
        "#2ecc71",
        "gap",
        "H3_dissociation_gap.png",
        "layers",
    )

    bands = report["band"]
    n_layers = len(metrics[0]["dH_query"])
    layers = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond, color in [("straightforward", "#3498db"), ("false_lead", "#e67e22")]:
        cond_data = [m for m in metrics if m["condition"] == cond]
        med = [
            float(np.median([m["dH_query"][i] for m in cond_data])) for i in layers
        ]
        ax.plot(layers, med, marker="o", label=cond, linewidth=2, markersize=5, color=color)
    ax.axvspan(
        bands[0] * (n_layers - 1),
        bands[1] * (n_layers - 1),
        alpha=0.18,
        color="gray",
        label=f"Workspace band [{bands[0]}, {bands[1]}]",
    )
    ax.axhline(
        report["theta"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f'theta={report["theta"]:.2f} nats',
    )
    ax.set_xlabel("Layer Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Excess Entropy ΔH (nats)", fontsize=12, fontweight="bold")
    ax.set_title("Entropy Collapse Curves by Condition (holdout)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "entropy_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for cond, color in [("straightforward", "#3498db"), ("false_lead", "#e67e22")]:
        xs, ys = [], []
        for m in metrics:
            if m["condition"] != cond or m["l_star"] is None:
                continue
            xs.append(m["l_star"])
            ys.append(m["oscillation"] or 0)
        ax.scatter(xs, ys, s=55, alpha=0.65, color=color, label=cond)
        if len(xs) >= 2:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xline = np.linspace(min(xs), max(xs), 50)
            ax.plot(xline, p(xline), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Commitment Layer ℓ*", fontsize=12, fontweight="bold")
    ax.set_ylabel("Oscillation Depth", fontsize=12, fontweight="bold")
    ax.set_title("Oscillation vs Commitment Layer", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "oscillation_vs_commitment.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, color in [("straightforward", "#3498db"), ("false_lead", "#e67e22")]:
        vals = [m["oscillation"] or 0 for m in metrics if m["condition"] == cond]
        ax.hist(
            vals,
            bins=np.arange(-0.5, max(vals) + 1.5, 1),
            alpha=0.55,
            label=cond,
            color=color,
            edgecolor="black",
        )
    ax.set_xlabel("Oscillation Depth", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count (holdout items)", fontsize=12, fontweight="bold")
    ax.set_title("H2 Detail: Oscillation Depth Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "H2_oscillation_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    rows = []
    for key in ["l_star", "oscillation", "gap"]:
        test = report["tests"][key]
        rows.append(
            {
                "Hypothesis": test["hypothesis"].split(":")[0].strip(),
                "n_pairs": test["n_pairs"],
                "n_nonzero": test["n_nonzero"],
                "Median_delta": f"{test['median_diff']:.1f}",
                "W_plus": test["W_plus"],
                "p_one_tailed": f"{test['p']:.6g}",
                "p_two_tailed": f"{test['p_two_sided']:.6g}",
                "Support": "YES" if test["p"] < 0.05 else "NO (p>=0.05)",
            }
        )
    df = pd.DataFrame(rows)
    (out / "table_hypotheses.txt").write_text(df.to_string(index=False) + "\n")

    by_pair = {}
    for m in metrics:
        by_pair.setdefault(m["pair_id"], {})[m["condition"]] = m
    lines = [
        "| Family | Pair | SF distractor_lead | FL distractor_lead |",
        "|---|---|---|---|",
    ]
    for pid, conds in sorted(by_pair.items()):
        if "straightforward" not in conds or "false_lead" not in conds:
            continue
        fam = conds["straightforward"]["family"]
        sf = conds["straightforward"]["distractor_lead_layers"]
        fl = conds["false_lead"]["distractor_lead_layers"]
        lines.append(f"| {fam} | {pid} | {sf} | {fl} |")
    (out / "table_distractor_temptation.md").write_text("\n".join(lines) + "\n")

    hm = Path("out/analysis_real/heatmaps")
    paper_hm = out / "heatmaps"
    paper_hm.mkdir(exist_ok=True)
    preferred = [
        "capital_cl_straightforward.png",
        "capital_cl_false_lead.png",
        "capital_eg_straightforward.png",
        "capital_eg_false_lead.png",
        "gp_horse_straightforward.png",
        "gp_horse_false_lead.png",
        "legs_spider_straightforward.png",
        "legs_spider_false_lead.png",
        "arith_c_straightforward.png",
        "arith_c_false_lead.png",
    ]
    for name in preferred:
        src = hm / name
        if src.exists():
            shutil.copy2(src, paper_hm / name)
    print("figures ready in", out)


if __name__ == "__main__":
    main()
