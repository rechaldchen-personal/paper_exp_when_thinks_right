# Discussion Section Outline (Section 6)

**Status**: Partially filled from Qwen3-1.7B GPU results (2026-07-17). Expand remaining subsections into prose for submission.  
**Target length**: 1500–2000 words

---

## 6. Discussion

### 6.1 Summary of Findings (200 words)

**Filled from GPU run**:
- **H1 (later commitment)**: Directional, **not significant** (p = 0.0625, n_pairs = 6; median Δ = +2.0 layers)
- **H2 (oscillation)**: **Supported** (p = 4.89×10⁻⁵, n_pairs = 32; median Δ = +2.0 changes; medians 3.0 vs 1.0)
- **H3 (dissociation gap)**: Directional, **not significant** (p = 0.0625, n_pairs = 6; median Δ = +4.0 layers)
- Hard controls: not evaluable in this holdout split (n=0 defined metrics)
- Overall: The strongest real-model signature is internal top-1 oscillation under false lead; delayed commitment and dissociation gap remain suggestive but underpowered

**Key sentence (updated)**:
> "False-lead prompts induce a robust internal revision signature in the J-space—increased top-1 oscillation—while delayed commitment and confidence–correctness gaps trend in the predicted direction but do not yet clear α=0.05 with the available complete pairs."

---

### 6.2 Interpretation: What This Means (400 words)

#### 6.2.1 The Dissociation Gap as a Hallucination Risk Marker

**What to emphasize**:
- Model achieves high confidence (low ΔH) before locking onto correct answer
- This "confidently-wrong window" is exactly where hallucinations happen
- Unprecedented visibility into pre-hallucinatory states

**Discussion points**:
- Connection to overconfidence biases in neural networks
- Alignment with human psychology: confidence ≠ correctness
- Practical application: real-time uncertainty flagging

**Writing template**:
> "The dissociation gap (ℓ* − ℓ_H) of X layers in false-lead prompts represents a previously invisible 'confidently-wrong window.' During this window, the model commits to a high-confidence state—achieving low excess entropy—before the correct answer becomes the top-1 prediction. This dissociation is absent in straightforward prompts (negative gap) and in hard controls (no collapse). This finding has direct implications for hallucination detection: a model's entropy collapse before answer commitment is a red flag for potential errors. Real-time monitoring of this gap could serve as an early-warning signal for unreliable model outputs."

---

#### 6.2.2 Oscillation as an Unsupervised Backtracking Signal

**What to emphasize**:
- Top-1 identity changes are internal analog of human revision
- Requires no explicit reasoning steps or intermediate labels
- Unsupervised detection of uncertainty

**Discussion points**:
- Parallels to human reading-time slowdowns on garden-path sentences
- Oscillation without explicit reasoning (emergent property of J-space)
- Could be useful for confidence scoring without auxiliary models

**Writing template**:
> "The oscillation signature—top-1 identity changes after confidence collapse—provides an unsupervised marker of internal revision, detectable without access to explicit reasoning steps or intermediate annotations. On false-lead prompts, the model's lens readout shows an average of X top-1 flips in the workspace band, absent in straightforward conditions. This pattern echoes human cognitive phenomena: just as humans show reading-time slowdowns on garden-path sentences (the classic behavioral signature of revision), LLMs show internal workspace oscillation. This convergence suggests that the J-space captures a dimension of model processing analogous to human uncertainty."

---

#### 6.2.3 Hard Controls Validate Specificity

**What to emphasize**:
- Effects are NOT just from difficulty
- False-lead signatures disappear when there's no tempting distractor
- Supports core hypothesis: backtracking is distraction-specific

**Discussion points**:
- Design robustness (controls work as intended)
- Confound analysis (difficulty ≠ false lead)
- Reproducibility (effects should hold across models)

---

### 6.3 Broader Context: Situating Within Gurnee et al. (2026) (300 words)

**What to write**: How this work extends Gurnee et al.'s global workspace findings.

**Key contrasts**:

| Gurnee et al. | This Work |
|---|---|
| Workspace band exists | When does commitment happen within the band? |
| Multi-hop rank climbs | What triggers rank climbs? (distraction vs. clarity) |
| Aggregate statistics | Per-prompt commitment trajectories |
| Architecture properties | Task difficulty modulates commitment |

**Writing template**:
> "Gurnee et al. (2026) established that a sparse J-space acts as a verbalizable workspace, with answers surfacing at intermediate layers. Our work asks a complementary question: *when and how* does commitment occur within this workspace, and whether difficulty or ambiguity leaves measurable traces. We find that false-lead prompts delay commitment by approximately X layers and induce internal oscillation, phenomena absent in straightforward or hard-control prompts. This suggests that the workspace not only *contains* answer representations but also exhibits *dynamics* reflecting the model's internal revision process. The dissociation gap—a novel finding—reveals that confidence and correctness can temporally dissociate, a phenomenon invisible to conventional output-level analysis. Together, these findings paint a picture of the J-space as not just a passive representational space but an active site of internal problem-solving."

---

### 6.4 Connection to Psycholinguistics & Human Cognition (250 words)

**What to emphasize**: Alignment with human parsing / revision literature.

**Key parallels**:
- Garden-path sentences → human reading-time slowdowns
- False-lead prompts → model oscillation (structural analog)
- Revision cost in humans → oscillation depth in models

**Discussion points**:
- Natural Stories external validity (when GPU data available)
- Correlation between model oscillation and human RT slowdowns
- Architectural differences (no explicit revisions in Transformers) but functional similarity

**Writing template**:
> "In human psycholinguistics, garden-path sentences induce reading-time slowdowns at the point of disambiguation—a behavioral signature of revision costs. We find a functional analog in LLMs: on garden-path prompts adapted from the Natural Stories corpus, the model's lens readout exhibits oscillation (top-1 identity changes). Preliminary correlation analysis [GPU phase] suggests that model oscillation depth correlates with human reading-time slowdowns (r ≈ ?), suggesting that LLMs exhibit cognitively-analogous revision signatures at the level of internal workspace dynamics. This convergence is striking given that Transformers lack explicit backtracking mechanisms: revision emerges as a distributed process across layers, invisible to output-level analysis."

---

### 6.5 Limitations & Future Work (300 words)

#### Limitations

**1. Small model size**
- Qwen 1.7B: open-weight advantage but smaller than frontier models
- Frontier models may show different commitment patterns
- Future: Replicate on larger open-weights models or via API access

**2. Single-token targets**
- J-lens vocabulary limitation
- Multi-token completions (reasoning steps) not captured
- Future: Extend to multi-token or logit-lens-only readouts

**3. English-dominant stimuli**
- All stimuli in English
- No cross-linguistic validation
- Future: Translate stimuli to other languages

**4. Correlational, not causal**
- We observe oscillation but don't manipulate what causes it
- Causal ablation (removing distractor representation) deferred
- Future: Ablation experiments on subset

**5. Theta set on same data**
- We use dev/holdout split, but ideal would be held-out test set
- Appendix shows sensitivity to theta, but still not independent
- Future: Collect larger dataset for true 3-way split

**6. Garden-path subset of Natural Stories**
- Only ~10–15 items curated from corpus
- Full corpus has ~100 stories; potential for larger study
- Future: Full Natural Stories analysis with larger sample

#### Future Directions

**Immediate (next semester)**:
- Replica on Qwen 4B and Llama family
- Logit-lens secondary readout validation
- Full Natural Stories correlation

**Medium-term (1–2 years)**:
- Causal ablation of distractor direction
- Multi-token extension
- Cross-lingual generalization

**Long-term (speculative)**:
- Real-time uncertainty quantification based on dissociation gap
- Pre-answer hallucination detection systems
- Architectural insights for improving model robustness

---

### 6.6 Practical Implications (250 words)

**What to emphasize**: Why this matters beyond academia.

**Applications**:

**1. Hallucination Detection**
- Dissociation gap as a risk marker
- Could be monitored at inference time
- No auxiliary models needed (already in lens readouts)

**2. Uncertainty Quantification**
- Model's internal confidence (ΔH) doesn't perfectly align with correctness (oscillation)
- Can use dissociation gap as uncertainty score
- Better than temper-scaling or other post-hoc methods (this is pre-output signal)

**3. Model Auditing & Safety**
- Inspect "confidently-wrong windows" for patterns
- Identify which prompt types are high-risk
- Improve data collection and training

**4. Interpretability Research**
- Commitment curves are interpretable proxy for internal reasoning
- Lens readouts reveal something about how models solve problems
- Helps us understand when and why models fail

**Writing template**:
> "Beyond theoretical interest, these commitment curves have practical implications. The dissociation gap identifies a 'hallucination-prone window' where models are most confident yet most vulnerable. In deployment, monitoring this gap in real-time could flag unreliable outputs before they reach users. The oscillation signature, visible in workspace dynamics without explicit reasoning steps, provides a lightweight uncertainty signal that doesn't require auxiliary classifiers or post-hoc calibration. For model auditors, commitment curves offer a new lens for understanding which prompt types cause internal revision—knowledge that could improve both training data curation and safety evaluations."

---

### 6.7 Concluding Remarks (150 words)

**What to write**: Tie it all together.

**Key take-home messages**:
1. Workspace isn't just representational; it's dynamic
2. Internal revision is detectable without explicit reasoning
3. Commitment curves are a new tool for interpretability & safety
4. More work needed on larger models and cross-lingual settings

**Final paragraph template**:
> "We began with a simple question: when does a language model know its answer? By instrumenting the J-space with layer-wise metrics, we find that 'knowing' is not instantaneous but a process—one that can be derailed by false leads and recovered through internal revision. The commitment curves we extract reveal dynamics invisible to conventional output-level analysis, offering a new window into how Transformers solve problems. As models scale and deployment stakes rise, tools for understanding and auditing internal processing become critical. The dissociation gap and oscillation signatures described here are promising candidates for real-time uncertainty quantification and hallucination detection, grounding interpretability in practical safety."

---

## Writing Checklist

- [ ] 6.1 Summary (all hypotheses stated)
- [ ] 6.2.1 Dissociation gap (hallucination angle)
- [ ] 6.2.2 Oscillation (revision angle)
- [ ] 6.2.3 Hard controls (specificity angle)
- [ ] 6.3 Gurnee et al. context (contribution clarity)
- [ ] 6.4 Psycholinguistics connection (external validity)
- [ ] 6.5 Limitations (honest assessment)
- [ ] 6.6 Practical implications (impact statement)
- [ ] 6.7 Concluding remarks (takeaway message)

---

## Style Notes

- Use "we find" / "our results show" (active voice)
- Avoid overclaiming ("reveals," "proves" → "suggests," "indicates")
- Link back to Gurnee et al. throughout
- Connect to human cognition where relevant
- Acknowledge limitations upfront
- End on future-work note (not closure)

---

## Length Target

- 6.1: 200 words
- 6.2: 400 words (3 subsections × 130 words each)
- 6.3: 300 words
- 6.4: 250 words
- 6.5: 300 words
- 6.6: 250 words
- 6.7: 150 words
- **Total: ~1850 words**

---

## Integration with GPU Results

Qwen3-1.7B primary run integrated (2026-07-17):
1. ✅ H1–H3 numbers filled in §6.1 and `RESULTS_TEMPLATE.md`
2. ⏸ Natural Stories correlation still pending
3. ⏸ Qwen 4B deferred
4. ✅ Figure paths: `out/figures/H1_*.png`, `H2_*.png`, `H3_*.png`, `entropy_curves.png`, `heatmaps/`
5. Remaining discussion subsections (6.2–6.7) still use writing templates—expand into final prose before submission
