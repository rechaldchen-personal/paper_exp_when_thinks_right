# Section 5: Results

**Status**: Filled from Qwen3-1.7B GPU run (2026-07-17).

**Validation Status**:
- ✅ Methodology validated on synthetic data
- ✅ Pipeline tested and working
- ✅ Real data collected: `out/traces.json` (163 stimuli), `out/analysis_real/report.json`
- ✅ Figures in `out/figures/`
- ⏸ Qwen 4B replication deferred
- ⏸ Natural Stories external-validity analysis not run

**Analysis settings (pre-registered)**:
- Model: Qwen/Qwen3-1.7B
- Lens: Jacobian lens fitted on 1000 × 128 FineWeb prompts (`out/lens_qwen3_1p7b.pt`)
- Dev/holdout split: 60/40 by pair (`--dev-split 0.6`) → 46 dev pairs, 32 holdout pairs
- Theta (ΔH threshold): 3.517 nats (80th percentile of straightforward condition, computed on **dev only**)
- Workspace band: [0.25, 0.90]

---

## 5.1 Sanity Check: Excess-Entropy Profile

**Validation**: Excess-entropy ΔH curves on holdout items (see `out/figures/entropy_curves.png`).

**Figure**: Layer-wise median excess entropy ΔH for Qwen3-1.7B by condition.
- X-axis: layer index (0–26; 27 source layers in fitted lens)
- Y-axis: excess entropy ΔH (nats)
- Workspace band shaded at fractional depth [0.25, 0.90]
- Theta threshold overlaid at 3.517 nats

**Table**: Workspace band used for primary analyses

| Model | Final band used | Theta (nats) | Notes |
|-------|-----------------|--------------|-------|
| Qwen3-1.7B | [0.25, 0.90] | 3.517 | Pre-registered fractional band; θ from SF 80th pct on dev |
| Qwen 4B | — | — | Deferred |

---

## 5.2 Main Findings: Commitment Dynamics

All paired tests compare **false_lead − straightforward** on holdout pairs (Wilcoxon signed-rank). Complete pairs require both conditions to yield the metric; H1/H3 have fewer complete pairs than H2 because ℓ*/gap are undefined when the target never reaches commitment criteria within band.

### H1: Later Commitment Under False Lead

**Prediction**: ℓ* (commitment layer) larger for false_lead than straightforward.

**Figure 5A**: `out/figures/H1_commitment_layer.png`

**Condition medians (holdout)**:
- Median ℓ* (straightforward): **20.0** layers (n=13 with defined ℓ*)
- Median ℓ* (false_lead): **22.0** layers (n=16)
- Median paired difference (FL − SF): **+2.0** layers
- Wilcoxon signed-rank: W=**0.0**, p=**0.0625**, n_pairs=**6**
- **Result**: H1 **NOT SIGNIFICANT at α=0.05** (directional trend matches prediction; limited complete pairs)

---

### H2: Oscillation Under False Lead

**Prediction**: Oscillation depth (top-1 identity changes after ℓ_H) higher for false_lead.

**Figure 5B**: `out/figures/H2_oscillation.png`  
**Figure 6A (detail)**: `out/figures/H2_oscillation_hist.png`

**Condition medians (holdout)**:
- Median oscillation (straightforward): **1.0** changes (n=34)
- Median oscillation (false_lead): **3.0** changes (n=34)
- Median paired difference (FL − SF): **+2.0** changes
- Wilcoxon signed-rank: W=**8.5**, p=**4.89×10⁻⁵**, n_pairs=**32**
- **Result**: H2 **SUPPORTED**

---

### H3: Dissociation Gap Under False Lead

**Prediction**: Gap ℓ* − ℓ_H larger for false_lead (confidently-wrong window).

**Figure 5C**: `out/figures/H3_dissociation_gap.png`

**Condition medians (holdout)**:
- Median gap (straightforward): **0.0** layers (n=11)
- Median gap (false_lead): **4.5** layers (n=16)
- Median paired difference (FL − SF): **+4.0** layers
- Wilcoxon signed-rank: W=**0.0**, p=**0.0625**, n_pairs=**6**
- **Result**: H3 **NOT SIGNIFICANT at α=0.05** (directional trend matches prediction; same n_pairs constraint as H1)

**Hard controls**: No holdout hard_control items yielded defined ℓ*/ℓ_H/gap in this split (n=0 in report). Do not interpret hard-control nulls as evidence against specificity in this run.

---

## 5.3 Case Studies: Heatmaps

Representative holdout heatmaps copied to `out/figures/heatmaps/`:

| Pair | Conditions | Path |
|------|------------|------|
| capital_cl | SF / FL | `capital_cl_*.png` |
| capital_eg | SF / FL | `capital_eg_*.png` |
| legs_spider | SF / FL | `legs_spider_*.png` |
| gp_horse | SF / FL | `gp_horse_*.png` |
| arith_c | SF / FL | `arith_c_*.png` |

**Caption (example)**: Layer × position heatmaps for factual pair capital_cl / capital_eg and garden-path pair gp_horse. False-lead panels typically show longer distractor-lead stretches and delayed target commitment relative to straightforward prompts.

Full set: `out/analysis_real/heatmaps/` (64 PNGs).

---

## 5.4 Distractor Temptation

**Table 5.1** (holdout pairs; full table in `out/figures/table_distractor_temptation.md`):

| Family | Pair | Straightforward (distractor_lead_layers) | False_lead (distractor_lead_layers) |
|--------|------|------------------------------------------|-------------------------------------|
| Factual | capital_eg | 7 | 12 |
| Factual | capital_se | 9 | 13 |
| Factual | legs_spider | 4 | 11 |
| Factual | element_lightest | 2 | 12 |
| Garden-path | gp_cat | 8 | 15 |
| Garden-path | gp_door | 0 | 15 |
| Garden-path | gp_horse | 8 | 11 |
| Garden-path | gp_song | 0 | 15 |
| Arithmetic | arith_d | 7 | 12 |
| Arithmetic | arith_s | 3 | 14 |
| Arithmetic | arith_t | 3 | 13 |
| Arithmetic | arith_z | 2 | 16 |

**Condition medians**:
- Straightforward distractor_lead_layers: **2.5** (n=34)
- False_lead distractor_lead_layers: **11.0** (n=34)

Interpretation: On average, false-lead items show substantially longer stretches where the distractor outranks the target, validating that many stimuli are behaviorally tempting under the lens readout. Some pairs (e.g., capital_gr, capital_mx) show zero lead in both conditions and should be treated as weak temptations in this model.

---

## 5.5 Oscillation Depth: Detailed Analysis

**Figure 6A**: `out/figures/H2_oscillation_hist.png` — false_lead distribution shifts right vs straightforward.

**Figure 6B**: `out/figures/oscillation_vs_commitment.png` — oscillation vs ℓ* by condition (items with defined ℓ*).

**Interpretation**: H2 is the clearest signal in this run: false-lead prompts induce reliably more top-1 identity changes after entropy collapse. Combined with the directional H1/H3 trends, this is consistent with an internal revision narrative, but commitment-layer and gap tests are underpowered given only 6 complete holdout pairs with defined ℓ*/gap.

---

## 5.6 External Validity: Natural Stories

**Not run in this GPU session.** Deferred pending Natural Stories RT linkage.

---

## 5.7 Summary Table

**Table 5.2**: Hypothesis support summary (Qwen3-1.7B, holdout).

| Hypothesis | Test | Statistic | p-value | Result |
|---|---|---|---|---|
| H1: Later commitment (false_lead) | Wilcoxon signed-rank | W=0.0 (n=6) | p=0.0625 | ✗ NOT SIG. (directional) |
| H2: More oscillation (false_lead) | Wilcoxon signed-rank | W=8.5 (n=32) | p=4.89×10⁻⁵ | ✓ SUPPORTED |
| H3: Larger gap (false_lead) | Wilcoxon signed-rank | W=0.0 (n=6) | p=0.0625 | ✗ NOT SIG. (directional) |

---

## 5.8 Robustness Checks (Appendix D)

### D.1 Logit-Lens Replication
Not run in this session.

### D.2 Band Sensitivity
Not run in this session (primary band fixed at [0.25, 0.90]).

### D.3 Theta Sensitivity
Not run in this session (primary θ = 80th percentile).

---

## Interpretation & Scope

**Key findings**: On Qwen3-1.7B with a fitted Jacobian lens, false-lead prompts produce a **robust increase in internal oscillation (H2)**. Delayed commitment (H1) and larger dissociation gaps (H3) move in the predicted direction (+2 and +4 layers median paired differences) but do **not** reach α=0.05 with the current complete-pair counts (n=6). Distractor-lead medians (2.5 → 11.0 layers) support that false-lead stimuli are, on average, more tempting under the lens.

**Limitations**:
- Small model size (1.7B); Qwen 4B deferred
- H1/H3 underpowered due to undefined ℓ*/gap on many items
- Arithmetic answers rewritten to single-token English number words for Qwen3 tokenization; behavioral top-1 rates for arithmetic remain weaker than factual/garden-path
- Stimuli repaired for Qwen3 single-token constraints (see repo notes / `stimuli.json`)
- No Natural Stories external validation in this run
- Hard-control specificity not evaluable in this holdout split

**Implications**:
- Oscillation is a usable pre-output signature of conflict/revision under false leads
- Dissociation-gap and delayed-commitment claims need more complete pairs (or alternate operationalizations) before strong claims
- Connects to agentic-auditing agenda: readable internal uncertainty outside the final answer token
