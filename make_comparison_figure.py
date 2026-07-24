#!/usr/bin/env python3
"""make_comparison_figure.py — one-off comparison figure: 1.7B vs 4B x jlens vs logit_lens.

Not part of the regular pipeline (unlike generate_figures.py, which makes the
single-model figures from out/analysis_real/). This exists specifically to
visualize the 4B replication finding: p-values for all six tests (primary +
confirmatory x H1/H2/H3), grouped by model and readout.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    ("1.7B\njlens", "out/analysis_real/report.json"),
    ("1.7B\nlogit_lens", "out/analysis_run2_logitlens/report.json"),
    ("4B\njlens", "out/analysis_4b/report.json"),
    ("4B\nlogit_lens", "out/analysis_4b_logitlens/report.json"),
]

METRICS = [
    ("l_star", "tests", "H1 primary"),
    ("l_star_2afc", "confirmatory_2afc", "H1 confirmatory"),
    ("oscillation", "tests", "H2 primary"),
    ("oscillation_2afc", "confirmatory_2afc", "H2 confirmatory"),
    ("gap", "tests", "H3 primary"),
    ("gap_2afc", "confirmatory_2afc", "H3 confirmatory (headline)"),
]

data = {label: json.load(open(path)) for label, path in RUNS}

fig, ax = plt.subplots(figsize=(11, 6))
n_runs = len(RUNS)
n_metrics = len(METRICS)
x = np.arange(n_metrics)
width = 0.8 / n_runs
colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]  # 1.7B greens, 4B reds

for i, (label, _) in enumerate(RUNS):
    pvals = [data[label][grp][key]["p"] for key, grp, _ in METRICS]
    neglog = [-np.log10(max(p, 1e-6)) for p in pvals]
    bars = ax.bar(x + i * width - 0.4 + width / 2, neglog, width,
                  label=label.replace("\n", " "), color=colors[i], alpha=0.85)
    for b, p in zip(bars, pvals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
               f"{p:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1,
          label="p = 0.05")
ax.set_xticks(x)
ax.set_xticklabels([m[2] for m in METRICS], fontsize=9)
ax.set_ylabel("-log10(p)  (higher = more significant)", fontsize=11, fontweight="bold")
ax.set_title("H1-H3 significance: 1.7B vs 4B, jlens vs logit_lens readout\n"
             "(H3 confirmatory was robust across every 1.7B sensitivity setting; "
             "does not replicate at 4B under any readout)",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = Path("out/figures/model_comparison_1p7b_vs_4b.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
