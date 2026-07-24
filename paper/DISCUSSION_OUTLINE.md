# Discussion Section Outline (Section 6)

**Status**: Filled from Qwen3-1.7B run 2 (2026-07-22) and the Qwen3-4B
replication (2026-07-24), superseding the run-1 and run-2-only drafts. Two
headline changes across these revisions: (1) H3 (dissociation gap) replaced
H2 (oscillation) as the lead finding once the confirmatory 2AFC test refuted
the "internal revision" reading; (2) **H3 itself does not replicate on
Qwen3-4B**, which is now the single most important qualifier in this section
and reframes the paper's contribution from "we found a mechanism" to "we
found a mechanism, and demonstrated that it does not generalize across model
scale the way a single-model study would assume." §6.2.2 (new) and §6.5–6.7
carry most of that reframing.
**Target length**: 2100–2400 words

---

## 6. Discussion

### 6.1 Summary of Findings (250 words)

**Filled from run 2 + 4B replication**:
- **H1 (later commitment)**: on 1.7B, primary null (p=0.109); confirmatory
  **supported** (p=0.0021, median Δ=+2.0 layers), robust across every θ/band
  setting, though markedly weaker under logit-lens (p=0.040). **Does not
  replicate on 4B** under either readout (p=0.334, p=0.461).
- **H2 (oscillation)**: primary supported on 1.7B at the pre-registered
  default only (p=0.0077) — null under θ=70/90, null under logit-lens
  (p=0.453), and null on 4B under both readouts. Confirmatory version is a
  **stable null across all four model×readout combinations** (p=0.27–0.46).
  Hard controls oscillate nearly as much as false-lead items on 1.7B.
- **H3 (dissociation gap)**: on 1.7B, the most robust result in the study —
  significant across every θ/band setting *and* under logit-lens (confirmatory
  p=0.0016 jlens, p=0.033 logit-lens). **Does not replicate on 4B** under
  either readout (p=0.369, p=0.434) — the same non-replication pattern as H1.
- Overall: on Qwen3-1.7B, the dissociation gap (H3) is a genuinely robust
  finding along every within-model axis tested (θ, band, lens method).
  Delayed target-vs-distractor commitment (H1) holds in the same sense.
  Neither generalizes to Qwen3-4B. Oscillation (H2) is not shown to reflect
  candidate revision at any setting or model tested — now the best-supported
  single conclusion in the paper.

**Key sentence**:
> "In Qwen3-1.7B, false-lead prompts reliably open a wider confidence–
> correctness dissociation window, a finding that survives every threshold,
> band, and lens-readout choice we tested. The same effect, measured
> identically, is absent in Qwen3-4B — evidence that this signature is
> scale- or capability-dependent rather than a fixed property of how these
> models process false-lead prompts, and a caution against generalizing
> single-model interpretability findings without testing them."

---

### 6.2 Interpretation: What This Means (700 words)

#### 6.2.1 The Dissociation Gap as a Hallucination Risk Marker — Within Qwen3-1.7B

**What to emphasize**:
- Model achieves high confidence (low ΔH) before locking onto the correct
  answer, and this window is measurably larger under false lead, in Qwen3-1.7B
- This "confidently-wrong window" is exactly where hallucinations happen
- The result is robust *within that model*: significant under both readouts
  and every pre-registered θ/band setting for the confirmatory metric — three
  independent robustness axes, all passed
- **This robustness claim must be stated with its scope attached going
  forward**: "robust" here means robust to analysis choices within Qwen3-1.7B,
  not robust across model scale — §6.2.2 shows those are different claims

**Writing template**:
> "Within Qwen3-1.7B, the dissociation gap (ℓ* − ℓ_H, measured against the
> target/distractor comparison) widens by a median of 4.0 layers under
> false-lead prompts (p = 0.0016), and this holds across every θ percentile,
> workspace-band width, and — critically — under an independent logit-lens
> readout that shares no fitted parameters with the primary Jacobian lens.
> During this window, the model has already committed to a high-confidence
> state before the correct answer has overtaken the tempting distractor.
> This is the most robust signature in the 1.7B analysis by a wide margin.
> It has direct implications for hallucination detection — a model's entropy
> collapse before answer commitment is a measurable, pre-output red flag —
> with one caveat that governs everything else in this Discussion: this
> robustness was established *within one model*. Whether it holds *across*
> models turns out to be a separate question with a different answer."

---

#### 6.2.2 Why the Effect Doesn't Generalize: Scale, Susceptibility, and a Caution for Single-Model Claims [NEW]

**This subsection is new** and is arguably the most important addition to
the Discussion. §6.2.1 establishes H3 as robust to every analysis choice
*within* Qwen3-1.7B. This subsection establishes that the same effect is
absent in Qwen3-4B, under the identical pre-registered pipeline, and argues
for what that means.

**What to emphasize**:
- H3 and H1 (confirmatory) are null on Qwen3-4B under both readouts — not
  marginally null, and not underpowered: median effect sizes are ~0 (vs. +2
  to +4 layers at 1.7B), and 4B's sample sizes are, if anything, larger
- Behavioral evidence points at *why*: Qwen3-4B is both more accurate (100%
  vs. 70.8% straightforward accuracy on arithmetic) and far less internally
  tempted by the false-lead framing (4.2% vs. 29.2% behavioral temptation
  rate on arithmetic; internal `distractor_lead_layers` gap shrinks from
  +4.5 to +1.5 layers) — roughly a threefold reduction in how much the
  manipulation perturbs the model's internal computation at all
- The natural reading: **the dissociation-gap mechanism may require a model
  susceptible enough to actually be internally tempted by the false-lead
  framing.** 1.7B was; 4B largely is not. This is a capability-gated effect,
  not a universal transformer property.
- This is not a failed replication to explain away — it is itself a
  substantive finding, and arguably a more important one than uniform
  replication would have been: it demonstrates concretely that a rigorously
  validated, multiply-robust single-model interpretability finding can fail
  to generalize one step up in scale, which is a caution the interpretability
  literature does not emphasize enough. Most single-model mechanistic
  findings are never tested this way.

**Discussion points**:
- Connects to the broader "faithfulness across scale" question in
  interpretability: robustness-to-analysis-choice (what §6.2.1 established)
  and robustness-to-scale are genuinely separate properties, and papers that
  establish only the former often implicitly claim the latter
- One data point (1.7B → 4B) cannot distinguish "the effect fades smoothly
  with scale/capability" from "1.7B has some specific property"; a third
  model size is the natural next test (§6.5)
- Practical implication for anyone building on dissociation-gap-style
  hallucination detectors: validate per-model before deploying, don't assume
  portability within a model family

**Writing template**:
> "The dissociation gap that is so robust in Qwen3-1.7B is entirely absent
> in Qwen3-4B: all six pre-registered tests (primary and confirmatory, H1
> and H3) that were significant in 1.7B are null in 4B, under both the
> primary and the logit-lens readout, with effect sizes near zero rather
> than merely non-significant. The explanation is not mysterious: Qwen3-4B
> is both more accurate on our stimuli and measurably less internally
> tempted by the false-lead framing — the median internal distractor-
> competition window shrinks roughly threefold, from +4.5 layers at 1.7B to
> +1.5 at 4B. If the manipulation barely perturbs the larger model's internal
> computation to begin with, there is little dissociation window left to
> detect. We read this as evidence that the effect is gated by a model's
> susceptibility to the false-lead framing rather than being a fixed
> property of how transformers process conflicting evidence in general. We
> report this prominently, rather than as a limitation buried in §6.5,
> because a within-model-robust effect that fails to generalize one step up
> in scale is itself an important finding about the fragility of
> single-model interpretability claims — one this study is positioned to
> demonstrate precisely because we tested for it rather than assuming it
> away."

---

#### 6.2.3 Oscillation: A Real Effect of Unresolved Origin

*(Renumbered from the previous draft's §6.2.2; content unchanged in
substance, now reinforced by the 4B data.)*

The original framing treated increased top-1 oscillation as direct evidence
of internal revision between the target and the tempting distractor. The
confirmatory test — read out over {target, distractor} only — does not
support that (p = 0.46 on 1.7B, stable across every sensitivity setting),
and hard-control items with no tempting distractor oscillate almost as much
as false-lead items. **The 4B data add a fourth independent line of
evidence**: the confirmatory metric is a stable null across all four
model×readout combinations (p range 0.27–0.46), the most consistent result
anywhere in this study.

**What to emphasize**:
- Global top-1 churn increases under false lead at the 1.7B/jlens/default-θ
  setting only — not nothing, but the narrowest possible evidentiary base
- The mechanism is open: churn among vocabulary tokens unrelated to either
  candidate is at least as plausible as "wavering between target and
  distractor," given four independent null results for the latter
- This is a case study in why the confirmatory-metric design was worth
  building, twice over: it caught an appealing but unsupported mechanistic
  story on 1.7B, and the 4B replication shows the same caution generalizes

**Writing template**:
> "Top-1 identity churn increases under false-lead prompts at the
> pre-registered 1.7B/jlens setting (median Δ=+0.5, p=0.0077), but every
> other test we ran — the confirmatory target-vs-distractor comparison, θ
> sensitivity, the logit-lens readout, the hard-control comparison, and the
> 4B replication — returns a null. We do not find evidence that this
> oscillation reflects internal revision between candidate answers at any
> setting; it more plausibly reflects general representational instability
> specific to a narrow analysis configuration on one model. We report the
> primary 1.7B effect because it is real and pre-registered, but withhold
> any 'internal backtracking' interpretation, which four independent checks
> now argue against."

---

#### 6.2.4 Hard Controls: Evidence Against H2 Specificity

*(Renumbered from the previous draft's §6.2.3; unchanged.)*

Hard-control oscillation (median 2.0, n=6 holdout, 1.7B) is comparable to
false-lead oscillation (2.5) and above straightforward (1.0) — undermining
rather than supporting oscillation as false-lead-specific. Hard-control gap
(median 1.5, n=2 holdout) is too small a sample to test but not obviously
larger than straightforward's, at least not inconsistent with H3 being
false-lead-specific within 1.7B. Small n means neither reading is
conclusive; expanding hard controls remains future work (§6.5).

---

### 6.3 Broader Context: Situating Within Gurnee et al. (2026) (350 words)

**Key contrasts**:

| Gurnee et al. | This Work |
|---|---|
| Workspace band exists | When does the target overtake a specific distractor within the band? |
| Multi-hop rank climbs | What triggers rank climbs? (distraction vs. clarity) |
| Aggregate statistics | Per-prompt commitment and dissociation trajectories |
| Architecture properties (one model class) | False-lead dissociation is capability-gated, tested explicitly across two model sizes |

**Writing template**:
> "Gurnee et al. (2026) established that a sparse J-space acts as a
> verbalizable workspace, with answers surfacing at intermediate layers,
> demonstrated primarily through aggregate, largely single-model-class
> statistics. Our work asks a complementary question — *when and how* does
> commitment occur within this workspace under conflicting evidence — and
> additionally asks whether the answer is a fixed property of the
> architecture or contingent on model scale. We find that in Qwen3-1.7B,
> false-lead prompts widen the confidence–correctness dissociation window by
> a median of 4.0 layers (p=0.0016, robust to every sensitivity setting and
> to an independent lens method) and delay target-over-distractor commitment
> by 2.0 layers (p=0.0021, likewise robust). Neither effect appears in
> Qwen3-4B under an identical pipeline. This suggests the workspace's
> *dynamics* under conflicting evidence are real, but — unlike the band's
> existence itself, which Gurnee et al. show to be consistent across model
> families — the false-lead dissociation dynamic we identify is
> capability-gated rather than a fixed architectural property. This is a
> meaningfully different kind of claim than 'the workspace exists,' and one
> that a single-model study would not have been positioned to make."

---

### 6.4 Connection to Psycholinguistics & Human Cognition (200 words)

**What survives**: garden-path stimuli showed the highest false-lead
temptation rate of the three families at 1.7B (66.7% vs. 25–29%; §5.4), and
the same qualitative pattern — garden-path most tempting, arithmetic
increasingly resistant — persists at 4B, where arithmetic temptation drops
to 4.2%. This is a behavioral fact about which stimulus types remain
effective at misleading a stronger model, independent of the internal-metric
non-replication.

**What must be dropped or heavily hedged**: any claim that oscillation depth
is a model-internal analog of human reading-time slowdowns — the mechanism
this rested on (candidate revision) is unsupported at every setting tested,
including 4B. The "r ≈ ?" Natural Stories correlation, if ever run, should
target the dissociation gap (H3), not oscillation — and should now be
understood as a Qwen3-1.7B-specific correlation to test, not a general model
property, given §6.2.2.

**Writing template**:
> "Garden-path stimuli produced the strongest false-lead temptation of the
> three families in both models we tested, though the absolute rate drops
> substantially for the more capable model — consistent with larger models
> being more robust to the kind of temporary misparse that reliably
> misleads humans and smaller models alike. We had hypothesized that
> model-internal oscillation would provide a mechanistic analog to human
> garden-path reading-time slowdowns; neither the confirmatory test on
> Qwen3-1.7B nor the Qwen3-4B replication supports this, and we withhold the
> claim. The dissociation gap remains the more promising candidate for a
> future correlational study against human reading times, with the added
> caveat that any such study should be run per-model rather than assumed to
> transfer."

---

### 6.5 Limitations & Future Work (450 words)

#### Limitations

**1. The dissociation-gap effect does not replicate at 4B — this is now a
finding, not an open gap**
- Previously framed as "single model, generalization untested." That framing
  is no longer accurate: generalization *was* tested, and failed, for the
  one step in scale available. The honest limitation is narrower and sharper:
  **one replication attempt, at one specific step in scale (≈2×
  compute), establishes non-replication at that step — not a monotonic
  scaling trend.** A third model size is needed to distinguish "the effect
  fades smoothly across this range" from "1.7B has some specific property a
  smooth story wouldn't predict."
- Future: replicate at a third size (e.g. Qwen3-8B or 14B) to distinguish
  these; ideally also a different model family, to separate "scale" from
  "this specific model family's training."

**2. Modest complete-pair counts**
- Primary H1/H3 rest on 14 complete holdout pairs at 1.7B (17–18 at 4B);
  confirmatory metrics on 18–26 across models. Larger than run 1's n=6, but
  a larger stimulus set would strengthen any future scale claim further.
- Future: expand the matched-pair set, particularly arithmetic and
  garden-path (behavioral accuracy still below factual's ceiling at 1.7B,
  though 4B is already near-ceiling on arithmetic).

**3. Single registered split per model**
- Results reflect one pre-registered 60/40 dev/holdout split (seed 42) per
  model; we did not resample the split itself.
- Future: report split-resampling stability as an additional robustness axis.

**4. Oscillation mechanism unresolved (not merely unreplicated)**
- Four independent checks (confirmatory metric, θ-sensitivity, logit-lens,
  4B) now argue against the candidate-revision reading, but none identify
  an alternative mechanism for the real 1.7B/jlens/default-θ effect.
- Future: extend the confirmatory readout beyond a 2-way comparison, or a
  causal ablation of the distractor direction to test whether it drives the
  churn at all.

**5. Single-token targets; English-only; correlational not causal; no
Natural Stories run** — unchanged from the previous draft; see prior
sessions' notes. If Natural Stories is run, target H3, not H2 (§6.4).

**6. Band identification not adopted**
- All four model×readout combinations suggest a band shifted later and
  wider than the pre-registered default, but the suggestion is partly driven
  by a next-token-accuracy signal that rises near-monotonically with depth
  and isn't specific to a genuine bounded workspace (`BAND_IDENTIFICATION_REPORT.md`).
  Not adopted; an open methodological question, not resolved here.

#### Future Directions

**Immediate**: a third model size to distinguish smooth scaling from a
1.7B-specific property; expand stimuli for power; logit-lens integration is
done (this session) — extend it to any future model added.

**Medium-term**: causal ablation of the distractor direction; multi-token
extension; cross-lingual generalization; a systematic study of *which*
capability threshold gates the dissociation-gap effect (accuracy? explicit
uncertainty calibration? something else?).

**Long-term (speculative)**: real-time uncertainty quantification based on
the dissociation gap, *conditional on per-model validation* given §6.2.2;
pre-answer hallucination detection systems; architectural insights into why
larger models resist the false-lead manipulation internally, not just
behaviorally.

---

### 6.6 Practical Implications (300 words)

**Applications**:

**1. Hallucination Detection — With a New Caveat**
- Dissociation gap as a risk marker — robust to every within-model check on
  1.7B — but §6.2.2 means it **cannot be assumed to transfer to a different
  model size without validation**. This is now the central practical caveat.
- No auxiliary models needed; monitorable at inference time — for models
  where the effect is confirmed to exist.

**2. Uncertainty Quantification**
- A model's internal confidence (ΔH collapse) can precede correctness by a
  measurable, false-lead-sensitive margin, in models susceptible to the
  manipulation — pre-output signal, unlike post-hoc calibration.

**3. Model Auditing & Safety — Validate Per-Model, Not Per-Family**
- The single clearest practical lesson from this study: a rigorously
  validated single-model interpretability signature (H3 on 1.7B, robust to
  θ, band, and lens method) failed to replicate one step up in scale within
  the same model family. Auditors should not assume a validated signature
  transfers across model sizes without re-testing it — the cost of testing
  (one GPU session per model, as done here) is small relative to the risk
  of deploying an audit signal that silently stops working at a different
  scale.

**4. A Cautionary Methodological Point (Oscillation)**
- The oscillation result remains a useful case study independent of scale:
  a single aggregate metric suggested a mechanistic story (revision) that a
  targeted confirmatory test — and, now, replication — did not support.

**Writing template**:
> "Beyond theoretical interest, this study's practical lesson is less 'the
> dissociation gap is a useful hallucination signal' and more 'validated
> interpretability signals need re-validation across scale before they are
> trusted as general auditing tools.' The dissociation gap survives every
> within-model robustness check we applied to Qwen3-1.7B — θ, band, and lens
> method — and would, on that evidence alone, look like a strong candidate
> for deployment. It is entirely absent in Qwen3-4B under the identical
> pipeline. For model auditors and safety researchers, the actionable
> takeaway is procedural: budget for a replication check across at least one
> step in model scale before treating any single-model interpretability
> signature as a general property of a model family."

---

### 6.7 Concluding Remarks (200 words)

**Key take-home messages**:
1. The confidence–correctness dissociation gap is real and robust to every
   within-model analysis choice tested in Qwen3-1.7B — the strongest
   single-model claim this paper can make
2. **The same effect does not replicate in Qwen3-4B**, and behavioral
   evidence suggests why: larger, more capable models are less internally
   susceptible to the false-lead manipulation to begin with
3. Internal oscillation increases under false lead only at the narrowest
   possible setting and is not shown to reflect candidate revision at any
   of four independent checks, including the 4B replication
4. Pre-registering a confirmatory readout and a cross-scale replication, with
   interpretation rules fixed in advance, is what let us report (2) and (3)
   honestly rather than as unqualified positive findings

**Final paragraph template**:
> "We began with a simple question: when does a language model know its
> answer? In Qwen3-1.7B, confidence and correctness measurably dissociate
> under false-lead framing, robust to every analysis choice we threw at it.
> But the same pipeline, applied identically to Qwen3-4B, found nothing —
> and the behavioral data suggest a coherent reason: the larger model is
> simply harder to fool, internally as well as behaviorally. We report this
> not as a disappointing footnote but as the paper's second real finding: a
> mechanistically clean, multiply-robust single-model interpretability
> result can fail to generalize one step up in scale, and the field should
> treat that possibility as a default hypothesis to test, not an edge case
> to assume away. The discipline of building a confirmatory readout, and
> then a cross-scale replication, and reporting both outcomes honestly, is,
> in our view, as much a contribution here as the dissociation gap itself."

---

## Writing Checklist

- [x] 6.1 Summary (all hypotheses, both models, both readouts)
- [x] 6.2.1 Dissociation gap, explicitly scoped to Qwen3-1.7B
- [x] 6.2.2 NEW: scale-dependence / non-replication at 4B, with mechanism
- [x] 6.2.3 Oscillation (reframed: unattributed churn, reinforced by 4B)
- [x] 6.2.4 Hard controls (reframed: evidence against H2 specificity)
- [x] 6.3 Gurnee et al. context (updated for cross-scale framing)
- [x] 6.4 Psycholinguistics connection (weakened, cross-scale caveat added)
- [x] 6.5 Limitations (non-replication reframed as a finding, not a gap)
- [x] 6.6 Practical implications (validate-per-model as the headline lesson)
- [x] 6.7 Concluding remarks (leads with 1.7B robustness, pivots to 4B null)

---

## Style Notes

- Use "we find" / "our results show" (active voice)
- Avoid overclaiming ("reveals," "proves" → "suggests," "indicates")
- **Never state an H1/H3 robustness claim without its model scope attached**
  — "robust in Qwen3-1.7B" is a materially different claim from "robust,"
  and conflating them is the single most important thing to avoid in the
  final prose
- When mentioning oscillation, always pair the primary result with the
  confirmatory null, the hard-control comparison, and the 4B null
- Link back to Gurnee et al. throughout
- Acknowledge limitations upfront
- End on future-work note (not closure)

---

## Length Target

- 6.1: 250 words · 6.2: 700 words (4 subsections) · 6.3: 350 words ·
  6.4: 200 words · 6.5: 450 words · 6.6: 300 words · 6.7: 200 words
- **Total: ~2450 words** (grew from ~1900 to accommodate the 4B section)

---

## Integration with GPU Results

Qwen3-1.7B run 2 (2026-07-22) + logit-lens (2026-07-22) + band identification
and Qwen3-4B replication (2026-07-24), all integrated:
1. ✅ H1–H3 numbers (primary + confirmatory, both models, both readouts)
   filled in §6.1 and `RESULTS_TEMPLATE.md`
2. ✅ Sensitivity results integrated (`SENSITIVITY_REPORT.md`)
3. ✅ Logit-lens comparison integrated (`LOGIT_LENS_REPORT.md`)
4. ✅ Hard-control comparison integrated (§6.2.4)
5. ✅ 4B replication integrated as a new major subsection (§6.2.2), not a
   footnote — this is the most consequential update in this revision
6. ✅ Band identification integrated (§6.5 limitation 6,
   `BAND_IDENTIFICATION_REPORT.md`)
7. ⏸ Natural Stories correlation still pending — retarget at H3 if run,
   understood as per-model given §6.2.2
8. ✅ Figure paths: `out/figures/H1_*.png`, `H2_*.png`, `H3_*.png`,
   `entropy_curves.png`, `heatmaps/`, `model_comparison_1p7b_vs_4b.png`
9. Remaining work: expand the writing templates above into final
   submission-ready prose (structured but still template-shaped)
