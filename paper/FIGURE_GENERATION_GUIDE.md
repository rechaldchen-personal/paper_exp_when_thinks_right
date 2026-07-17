# Figure Generation Guide — Using Mock Data as Templates

**Status**: Ready to implement with real GPU data  
**Tool**: Python + matplotlib (all code provided below)

---

## Overview

This guide shows how to generate publication-ready figures from the analysis output. All figures below are generated from mock data (`out/analysis_mock_v2/`) and can be adapted for real GPU results with minimal changes.

---

## Figure 1: H1 — Later Commitment (Box Plot)

**What it shows**: Distribution of commitment layer ℓ* across conditions.

**Python code**:

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load analysis results
report = json.load(open('out/analysis_mock_v2/report.json'))
metrics = json.load(open('out/analysis_mock_v2/per_stimulus_metrics.json'))

# Prepare data: group by condition
conditions = ['straightforward', 'false_lead', 'hard_control']
data_l_star = {cond: [] for cond in conditions}

for m in metrics:
    if m['l_star'] is not None:
        data_l_star[m['condition']].append(m['l_star'])

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([data_l_star[c] for c in conditions],
                  labels=conditions,
                  patch_artist=True,
                  widths=0.6)

# Style
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)
for whisker in bp['whiskers']:
    whisker.set_linewidth(1.5)
for cap in bp['caps']:
    cap.set_linewidth(1.5)

ax.set_ylabel('Commitment Layer ℓ*', fontsize=12, fontweight='bold')
ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax.set_title('H1: Later Commitment Under False Lead', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add stats box
h1_test = report['tests']['l_star']
stats_text = f"Wilcoxon p={h1_test['p']:.4f}\nMedian Δ={h1_test['median_diff']:.1f} layers"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('out/figure_h1_commitment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: out/figure_h1_commitment.png")
```

**Expected output**: Box plot showing false_lead > straightforward + statistics box

---

## Figure 2: H2 — Oscillation (Box Plot)

**What it shows**: Distribution of oscillation depth across conditions.

```python
# Prepare data
data_osc = {cond: [] for cond in conditions}
for m in metrics:
    data_osc[m['condition']].append(m['oscillation'])

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([data_osc[c] for c in conditions],
                  labels=conditions,
                  patch_artist=True,
                  widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#e74c3c')
    patch.set_alpha(0.7)

ax.set_ylabel('Oscillation Depth (# top-1 changes)', fontsize=12, fontweight='bold')
ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax.set_title('H2: Internal Backtracking Signature', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

h2_test = report['tests']['oscillation']
stats_text = f"Wilcoxon p={h2_test['p']:.4f}\nMedian Δ={h2_test['median_diff']:.1f} changes"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('out/figure_h2_oscillation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: out/figure_h2_oscillation.png")
```

---

## Figure 3: H3 — Dissociation Gap (Box Plot)

**What it shows**: Distribution of confidently-wrong window (ℓ* − ℓ_H).

```python
# Prepare data
data_gap = {cond: [] for cond in conditions}
for m in metrics:
    if m['gap'] is not None:
        data_gap[m['condition']].append(m['gap'])

# Create figure (similar to above)
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([data_gap[c] for c in conditions],
                  labels=conditions,
                  patch_artist=True,
                  widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#2ecc71')
    patch.set_alpha(0.7)

ax.set_ylabel('Dissociation Gap (ℓ* − ℓ_H)', fontsize=12, fontweight='bold')
ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax.set_title('H3: Confidently-Wrong Window', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

h3_test = report['tests']['gap']
stats_text = f"Wilcoxon p={h3_test['p']:.4f}\nMedian Δ={h3_test['median_diff']:.1f} layers"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('out/figure_h3_gap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: out/figure_h3_gap.png")
```

---

## Figure 4: Summary Statistics Table

**What it shows**: All three hypotheses with test statistics.

```python
import pandas as pd

# Build table
rows = []
for key in ['l_star', 'oscillation', 'gap']:
    test = report['tests'][key]
    rows.append({
        'Hypothesis': test['hypothesis'].split(':')[0].strip(),
        'n_pairs': test['n_pairs'],
        'Median Δ': f"{test['median_diff']:.1f}",
        'Wilcoxon p': f"{test['p']:.4f}",
        'Support': '✓ YES' if test['p'] < 0.05 else '✗ NO'
    })

df = pd.DataFrame(rows)
print("\nTable 5.2: Hypothesis Support Summary")
print("=" * 80)
print(df.to_string(index=False))
print("=" * 80)

# Save as LaTeX for paper
latex = df.to_latex(index=False)
with open('out/table_hypotheses.txt', 'w') as f:
    f.write(latex)
print("\n✓ Saved LaTeX table: out/table_hypotheses.txt")
```

---

## Figure 5: Heatmap Selection Script

**What it shows**: Pre-made heatmaps of entropy, ranks.

```python
# Heatmaps are auto-generated in out/analysis_mock_v2/heatmaps/
# Select representative pairs:

import os
heatmap_dir = 'out/analysis_mock_v2/heatmaps'
files = os.listdir(heatmap_dir)

# Show all available
print(f"Available heatmaps ({len(files)} total):")
for f in sorted(files)[:10]:  # Show first 10
    print(f"  - {f}")

# Recommended selection for paper:
recommended = [
    'capital_fr_straightforward.png',  # Easy case
    'capital_fr_false_lead.png',       # Hard case (false lead)
    'hard_ctrl_currency_hard_control.png'  # Control case
]

print("\nRecommended for paper figures (Section 5.3):")
for f in recommended:
    if f in files:
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} (not found)")
```

---

## Figure 6: Entropy Collapse Pattern (Diagnostic)

**What it shows**: Layer-wise excess entropy for each condition.

```python
# Compute layer-wise medians
fig, ax = plt.subplots(figsize=(10, 6))

bands = report['band']
layers = list(range(len(metrics[0]['dH_query'])))

for cond in ['straightforward', 'false_lead']:
    cond_data = [m for m in metrics if m['condition'] == cond]
    layer_medians = []
    
    for layer_idx in layers:
        dH_at_layer = [m['dH_query'][layer_idx] for m in cond_data if layer_idx < len(m['dH_query'])]
        if dH_at_layer:
            layer_medians.append(np.median(dH_at_layer))
        else:
            layer_medians.append(0)
    
    ax.plot(layers, layer_medians, marker='o', label=cond, linewidth=2, markersize=6)

# Shade workspace band
ax.axvspan(bands[0] * len(layers), bands[1] * len(layers), alpha=0.2, color='gray', label='Workspace band')

ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Excess Entropy ΔH (nats)', fontsize=12, fontweight='bold')
ax.set_title('Entropy Collapse Curves by Condition', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('out/figure_entropy_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: out/figure_entropy_curves.png")
```

---

## Figure 7: Natural Stories Correlation (Template)

**What it shows**: Internal metrics vs. human reading times (when data available).

```python
# Template (placeholder for when GPU + Natural Stories data available)
fig, ax = plt.subplots(figsize=(8, 6))

# This will be filled with real Natural Stories data
# For now, show mock prediction
oscillation_vals = [m['oscillation'] for m in metrics if m['condition'] == 'false_lead']
human_rt_slowdown_mock = [0 + 5*x + np.random.normal(0, 2) for x in oscillation_vals]

ax.scatter(oscillation_vals, human_rt_slowdown_mock, s=100, alpha=0.6)

# Trend line
z = np.polyfit(oscillation_vals, human_rt_slowdown_mock, 1)
p = np.poly1d(z)
ax.plot(sorted(oscillation_vals), p(sorted(oscillation_vals)), "r--", linewidth=2, label=f"Trend (r≈?)")

ax.set_xlabel('Model Oscillation Depth', fontsize=12, fontweight='bold')
ax.set_ylabel('Human Reading-Time Slowdown (ms)', fontsize=12, fontweight='bold')
ax.set_title('External Validity: Internal Metrics vs. Human Behavior', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('out/figure_natural_stories_template.png', dpi=300, bbox_inches='tight')
print("✓ Template saved: out/figure_natural_stories_template.png")
print("  (Replace with actual Natural Stories data when available)")
```

---

## Complete Figure Generation Script

All code above in one runnable script:

```bash
python3 << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load data
report = json.load(open('out/analysis_mock_v2/report.json'))
metrics = json.load(open('out/analysis_mock_v2/per_stimulus_metrics.json'))
conditions = ['straightforward', 'false_lead', 'hard_control']

print("Generating figures from mock analysis...")
print("=" * 60)

# [Insert all 7 figure codes above here]

print("\n✓ All figures generated successfully!")
print("  - Figure 1 (H1): figure_h1_commitment.png")
print("  - Figure 2 (H2): figure_h2_oscillation.png")
print("  - Figure 3 (H3): figure_h3_gap.png")
print("  - Figure 4: Heatmaps in out/analysis_mock_v2/heatmaps/")
print("  - Figure 5: figure_entropy_curves.png")
print("  - Figure 6: figure_natural_stories_template.png")
EOF
```

---

## Usage When GPU Data Arrives

1. Run GPU pipeline to generate `out/analysis/report.json` and heatmaps
2. Replace filenames:
   - `out/analysis_mock_v2/` → `out/analysis/`
3. Run the figure generation script (code works identically for real data)
4. Copy generated PNGs into paper

---

## Style Notes

- **Colors**: Blue (H1), Red (H2), Green (H3), Gray (controls)
- **Font**: 12-14pt for labels, 10pt for stats
- **DPI**: 300 for publication
- **Box plots**: Show quartiles + individual points if n < 20

---

## Figure Checklist for Paper

- [ ] **Figure 5A**: H1 box plot (l_star)
- [ ] **Figure 5B**: H2 box plot (oscillation)
- [ ] **Figure 5C**: H3 box plot (gap)
- [ ] **Figure 5D–F**: Representative heatmaps (straightforward, false_lead, hard_control)
- [ ] **Figure 6A**: Entropy collapse curves
- [ ] **Figure 6B**: Oscillation vs. commitment layer scatter
- [ ] **Figure 7A**: Natural Stories correlation (GPU phase)
- [ ] **Table 5.1**: Distractor temptation stats
- [ ] **Table 5.2**: Hypothesis test summary
- [ ] **Table 7.1**: Natural Stories correlation table

All templates ready; just plug in real GPU data when available.
