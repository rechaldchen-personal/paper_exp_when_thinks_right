# Section 5: Results

⚠️ **IMPORTANT**: This is a TEMPLATE awaiting GPU data. Do not interpret empty fields as results. All numbers marked `?` will be filled from real Qwen3-1.7B traces after GPU phase.

**Validation Status**: 
- ✅ Methodology validated on synthetic data
- ✅ Pipeline tested and working
- ⏳ Real data validation pending GPU phase
- ❌ Do NOT cite these sections as confirmed results until GPU data is obtained

---

## 5.1 Sanity Check: Excess-Entropy Profile

**Validation**: Replicate Gurnee et al. (2026) excess-entropy curve.

**Figure**: Layer-wise excess entropy ΔH (median over all prompts) for Qwen 1.7B.
- X-axis: layer index (0–100 reindexing)
- Y-axis: excess entropy ΔH (nats)
- Expected shape: low in early layers, peak in workspace [0.25–0.90], decline in late layers
- Comparison: overlay kurtosis curve (Gurnee et al. Fig 28) to validate band identification

**Table**: Workspace band boundaries (per-model, per-criterion)
| Model | Excess-kurtosis band | Autocorr band | Accuracy band | Final band |
|-------|----------------------|---|---|---|
| Qwen 1.7B | [?, ?] | [?, ?] | [?, ?] | [?, ?] |
| Qwen 4B | [?, ?] | [?, ?] | [?, ?] | [?, ?] |

---

## 5.2 Main Findings: Commitment Dynamics

### H1: Later Commitment Under False Lead

**Prediction**: ℓ* (commitment layer) larger for false_lead than straightforward.

**Figure 5A**: Box plot of ℓ* by condition.
- X-axis: condition (straightforward, false_lead, hard_control)
- Y-axis: commitment layer ℓ*
- Points: individual pairs (n=15 pairs, each with 2 conditions)
- Error bars: quartiles; overlay individual pairs as lines to show pairing

**Statistics**:
- Median ℓ* (straightforward): ? layers
- Median ℓ* (false_lead): ? layers
- Difference: ? layers
- Wilcoxon signed-rank: W=?, p=?
- **Result**: H1 **SUPPORTED / REJECTED**

---

### H2: Oscillation Under False Lead

**Prediction**: Oscillation depth (top-1 identity changes after ℓ_H) higher for false_lead.

**Figure 5B**: Oscillation counts by condition.
- X-axis: condition
- Y-axis: oscillation depth (count of top-1 changes)
- Similar layout to 5A
- Many straightforward pairs will have oscillation=0

**Statistics**:
- Median oscillation (straightforward): ? changes
- Median oscillation (false_lead): ? changes
- Difference: ? changes
- Wilcoxon signed-rank: W=?, p=?
- **Result**: H2 **SUPPORTED / REJECTED**

---

### H3: Dissociation Gap Under False Lead

**Prediction**: Gap ℓ* − ℓ_H larger for false_lead (confidently-wrong window).

**Figure 5C**: Dissociation gap by condition.
- Similar box-plot layout
- Note: hard_control usually has gap=None (target never achieves low entropy); report as 0 or exclude from stats

**Statistics**:
- Median gap (straightforward): ? layers
- Median gap (false_lead): ? layers
- Difference: ? layers
- Wilcoxon signed-rank: W=?, p=?
- **Result**: H3 **SUPPORTED / REJECTED**

---

## 5.3 Case Studies: Heatmaps

**Figure 5D–F**: Per-pair (layer × position) heatmaps for one representative false_lead pair (e.g., "capital_fr").

3-panel heatmap per condition:
- **Panel 1**: Excess entropy ΔH (layer × position)
  - Color: viridis (yellow=high, purple=low)
  - Expect: early dip on distractor position, late dip on query position (false_lead)
  
- **Panel 2**: log₁₀(target_rank + 1)
  - Color: magma_r (yellow=top-ranked, purple=low-ranked)
  - Expect: target climbs to top-1 late (false_lead vs straightforward)
  
- **Panel 3**: log₁₀(distractor_rank + 1)
  - Similar layout
  - Expect: distractor peaks early (false_lead)

**Caption**: "Layer × position heatmaps for factual pair capital_fr (Paris vs. Amsterdam). Left: straightforward condition shows early entropy collapse at query position; Right: false_lead condition shows two-phase entropy profile with distractor dip, then recovery, then target dip."

---

## 5.4 Distractor Temptation

**Table 5.1**: Quantifying how much the distractor tempted the model.

| Family | Pair | Straightforward (distractor_lead_layers) | False_lead (distractor_lead_layers) |
|--------|------|---|---|
| Factual | capital_fr | ? | ? |
| Factual | capital_jp | ? | ? |
| ... | ... | ... | ... |
| Arithmetic | arith_a | ? | ? |
| ... | ... | ... | ... |

Interpretation:
- Straightforward: low distractor_lead (target always favored)
- False_lead: moderate-to-high (distractor briefly leads)
- Validates stimulus design: distractors truly tempting

---

## 5.5 Oscillation Depth: Detailed Analysis

**Figure 6A**: Oscillation depth (histogram or KDE) by condition.
- Overlay: straightforward (blue) vs. false_lead (orange)
- False_lead should shift right (more oscillations)
- Many straightforward items at 0; false_lead spread across 0–5+

**Figure 6B**: Relationship between oscillation depth and ℓ*.
- X-axis: ℓ* (commitment layer)
- Y-axis: oscillation_depth
- Color: condition (straightforward=blue, false_lead=orange)
- Expectation: false_lead at later layers (right) with more oscillation (higher)
- Trend line per condition

**Interpretation**: Does internal revision (oscillation) appear *after* the model first got confident? If so, supports the revision narrative.

---

## 5.6 External Validity: Natural Stories

(Only if GPU data available)

**Figure 7A**: Scatter plot of internal metrics vs. human reading times.
- X-axis: oscillation_depth or dissociation_gap (internal)
- Y-axis: human RT slowdown (ms) at disambiguation region
- Points: individual Natural Stories sentences (n~8)
- Trend line + R² + p-value

**Expected correlation**: r > 0.4, p < 0.05 (supporting hypothesis 7.4)

**Table 7.1**: Correlation matrix (oscillation, gap, ℓ*, ℓ_H) × human RT.

---

## 5.7 Summary Table

**Table 5.2**: Hypothesis support summary.

| Hypothesis | Test | Statistic | p-value | Result |
|---|---|---|---|---|
| H1: Later commitment (false_lead) | Wilcoxon signed-rank | W=? | p=? | **✓ SUPPORTED / ✗ REJECTED** |
| H2: More oscillation (false_lead) | Wilcoxon signed-rank | W=? | p=? | **✓ SUPPORTED / ✗ REJECTED** |
| H3: Larger gap (false_lead) | Wilcoxon signed-rank | W=? | p=? | **✓ SUPPORTED / ✗ REJECTED** |

---

## 5.8 Robustness Checks (Appendix D)

### D.1 Logit-Lens Replication
Repeat all main figures with logit-lens (identity transport) instead of J-lens.
- Figure D1–D3: Same structure as Figures 5A–C
- Report: Do H1–H3 hold under logit-lens? (Expected: yes, if J-lens findings are robust)

### D.2 Band Sensitivity
Rerun 03_analyze.py with alternative band definitions: [0.15, 0.95], [0.30, 0.85].
- Table D1: Paired test results (median diff, p-value) for each band choice
- Expected: qualitative findings hold across reasonable band choices

### D.3 Theta Sensitivity
Rerun with --theta-pct 70, 80, 90.
- Table D2: Do H1–H3 results depend heavily on θ choice?
- Expected: robust to reasonable θ choices

---

## Interpretation & Scope

(To be written after data available; sketch below)

**Key findings**: The three hypotheses **[SUMMARY: which supported, which not]** suggest that...

[Space for narrative on what commitment curves reveal about internal processing, alignment with human-like revision, implications for uncertainty quantification, etc.]

**Limitations**:
- Small model size (1.7B); frontier models may differ
- Single-token targets only; multi-token extensions unclear
- Mostly English stimuli; cross-lingual generalization open
- Correlational analysis; causal ablations on subset only

**Implications**:
- Hallucination detection: low ΔH at emission → model never committed
- Uncertainty flagging: pre-answer oscillation → model unsure
- Connects to broader agentic-auditing agenda (pre-output signal readable outside answer channel)
