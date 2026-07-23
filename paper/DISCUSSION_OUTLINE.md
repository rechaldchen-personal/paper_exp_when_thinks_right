# Discussion Section Outline (Section 6)

**Status**: Filled from Qwen3-1.7B run 2 (2026-07-22), superseding the run-1
draft. The headline changed: H3 (dissociation gap) is now the lead finding,
and §6.2.2/§6.4 are substantially rewritten because the confirmatory 2AFC
test (`PRE_REGISTRATION_AMENDMENT.md`) refutes the "internal revision"
reading of oscillation that the original draft was built around.
**Target length**: 1500–2000 words

---

## 6. Discussion

### 6.1 Summary of Findings (200 words)

**Filled from run 2**:
- **H1 (later commitment)**: primary null (p=0.109, n=14; median Δ=0.0
  layers); confirmatory **supported** (p=0.0021, n=19; median Δ=+2.0 layers),
  robust across all sensitivity settings
- **H2 (oscillation)**: primary supported at the pre-registered default
  (p=0.0077, n=28; median Δ=+0.5) but **not robust to θ** (null at θ70/θ90);
  confirmatory **null and stable** (p=0.46, consistent p≈0.46–0.62 across
  every sensitivity setting); hard controls oscillate nearly as much as
  false-lead items (median 2.0 vs 2.5, both above straightforward's 1.0)
- **H3 (dissociation gap)**: primary supported at the default (p=0.0112,
  n=14) but θ-fragile; confirmatory **supported and robust everywhere**
  (p=0.0016, n=18; median Δ=+4.0 layers) — the strongest and most stable
  result in the study
- Overall: the dissociation gap (H3) is the headline finding. Delayed
  target-vs-distractor commitment (H1) holds in the same robust,
  confirmatory sense. Oscillation (H2) is a real effect of unresolved origin,
  not confirmed evidence of revision between candidate answers.

**Key sentence**:
> "False-lead prompts reliably open a wider confidence–correctness
> dissociation window before the model reaches the target answer, a finding
> that holds across every workspace-band and threshold setting we tested.
> Top-1 oscillation also increases under false lead, but a targeted
> confirmatory test shows this is not attributable to wavering between the
> candidate answers specifically, so we report it as unattributed churn
> rather than internal revision."

---

### 6.2 Interpretation: What This Means (450 words)

#### 6.2.1 The Dissociation Gap as a Hallucination Risk Marker

**What to emphasize**:
- Model achieves high confidence (low ΔH) before locking onto the correct
  answer, and this window is measurably larger under false lead
- This "confidently-wrong window" is exactly where hallucinations happen
- The result is robust: significant under both readouts, and under every
  pre-registered θ/band sensitivity setting for the confirmatory metric

**Discussion points**:
- Connection to overconfidence biases in neural networks
- Alignment with human psychology: confidence ≠ correctness
- Practical application: real-time uncertainty flagging
- Why this is the strongest claim in the paper: unlike H1/H2, H3's
  confirmatory metric never loses significance across the sensitivity grid,
  and its primary metric never disagrees in direction

**Writing template**:
> "The dissociation gap (ℓ* − ℓ_H), measured against the target/distractor
> comparison, widens by a median of 4.0 layers under false-lead prompts
> (p = 0.0016), and this holds across every θ percentile (70th/80th/90th)
> and workspace-band width we pre-registered for sensitivity testing. During
> this window, the model has already committed to a high-confidence state —
> low excess entropy — before the correct answer has overtaken the tempting
> distractor. This dissociation is smaller in straightforward prompts and is
> the single most robust signature in this study. It has direct implications
> for hallucination detection: a model's entropy collapse before answer
> commitment is a measurable, pre-output red flag, and — unlike the
> oscillation signature discussed next — this one survives every robustness
> check we ran."

---

#### 6.2.2 Oscillation: A Real Effect of Unresolved Origin

**This subsection replaces the original "oscillation as backtracking
signal" framing.** The original draft treated increased top-1 oscillation as
direct evidence of internal revision between the target and the tempting
distractor. Run 2's confirmatory test — read out over {target, distractor}
only, rather than the full vocabulary — was added specifically to test that
claim, and it does not support it (p = 0.46, stable across every sensitivity
setting), while hard-control items with no tempting distractor at all
oscillate almost as much as false-lead items (median 2.0 vs 2.5). The
primary metric's own significance is also θ-fragile (null at θ70/θ90),
which independently weakens confidence in it as a robust phenomenon.

**What to emphasize instead**:
- Global top-1 churn genuinely increases under false lead by the
  pre-registered metric, at the pre-registered θ — this is not nothing
- But the mechanism is open: churn among vocabulary tokens unrelated to
  either candidate answer is at least as plausible an explanation as
  "wavering between target and distractor," given the confirmatory null
- Hard controls oscillating nearly as much as false-lead items suggests the
  effect may track general representational instability or difficulty
  rather than false-lead-specific conflict
- This is a case study in why the confirmatory-metric design was worth
  building: a plausible, appealing mechanistic story (candidate revision)
  did not survive a direct test, and the pre-registered decision rule
  (amendment §5) caught it before it became an overclaim

**Discussion points**:
- What "unattributed churn" could mean mechanistically: attention pattern
  instability, competition among near-synonyms or distractor-adjacent
  tokens not scored here, or general effects of prompt length/complexity
  (false-lead prompts are longer)
- Why the parallel to human garden-path revision (§6.4 in the original
  draft) is now a much weaker claim than previously framed
- What would resolve this: extending the 2AFC comparison to a wider
  candidate set, or a causal ablation of the distractor direction (future
  work, §6.5)

**Writing template**:
> "Top-1 identity changes after confidence collapse increase under
> false-lead prompts by the pre-registered global metric (median Δ=+0.5,
> p=0.0077), but a confirmatory test restricted to the target and distractor
> tokens specifically — designed to test whether this churn reflects
> wavering between the two candidate answers — is null and stable across
> every sensitivity setting we ran (p≈0.46–0.62). Hard-control items, which
> contain no tempting distractor at all, oscillate almost as much as
> false-lead items (median 2.0 vs. 2.5 changes). Taken together, we do not
> find evidence that this oscillation reflects internal revision between
> candidate answers; it more plausibly reflects general representational
> instability that is not specific to the false-lead manipulation. We report
> the primary effect because it is real and pre-registered, but we
> explicitly withhold the 'internal backtracking' interpretation that a
> single-metric analysis would have invited."

---

#### 6.2.3 Hard Controls: Evidence Against H2 Specificity, Not Evidence For It

**This subsection is reframed from the original draft**, which anticipated
hard controls validating specificity (no oscillation without a tempting
distractor). Run 2's hard controls show the opposite pattern for H2 and are
inconclusive (n=2) for H3.

**What to emphasize**:
- Hard-control oscillation (median 2.0, n=6 holdout) is comparable to
  false-lead oscillation (2.5) and above straightforward (1.0) — this
  undermines rather than supports treating oscillation as false-lead-specific,
  and is the second independent piece of evidence (alongside the confirmatory
  null) against the revision interpretation
- Hard-control gap (median 1.5, n=2 holdout) is too small a sample to test,
  but is not obviously larger than straightforward's gap (1.0), which is at
  least not inconsistent with H3 being false-lead-specific
- Small n (2–6) means neither reading should be treated as conclusive;
  expanding hard controls further is listed under Future Work

**Discussion points**:
- Design robustness: the controls behaved informatively even without
  reaching significance — they changed which claims we're willing to make
- This is a case where a null/ambiguous control result did real
  evidentiary work rather than being a formality

---

### 6.3 Broader Context: Situating Within Gurnee et al. (2026) (300 words)

**What to write**: How this work extends Gurnee et al.'s global workspace
findings, updated to the run-2 numbers and the narrower H1/H2 claims.

**Key contrasts**:

| Gurnee et al. | This Work |
|---|---|
| Workspace band exists | When does the target overtake a specific distractor within the band? |
| Multi-hop rank climbs | What triggers rank climbs? (distraction vs. clarity) |
| Aggregate statistics | Per-prompt commitment and dissociation trajectories |
| Architecture properties | False-lead framing modulates confidence–correctness dissociation |

**Writing template**:
> "Gurnee et al. (2026) established that a sparse J-space acts as a
> verbalizable workspace, with answers surfacing at intermediate layers. Our
> work asks a complementary question: *when and how* does commitment occur
> within this workspace, and whether a false-lead framing leaves measurable
> traces beyond the final answer. We find that false-lead prompts widen the
> confidence–correctness dissociation window by a median of 4.0 layers
> (confirmatory metric, p=0.0016, robust to every sensitivity setting
> tested) and delay the point at which the target specifically overtakes a
> tempting distractor by a median of 2.0 layers (p=0.0021, likewise robust).
> Both effects hold even though a naively appealing companion claim —
> that the model's oscillating top-1 identity reflects active revision
> between the two candidates — did not survive a targeted confirmatory
> test. This suggests the workspace's *dynamics* under conflicting evidence
> are real but more subtle than a single aggregate metric would suggest:
> confidence and correctness can dissociate measurably, while the mechanism
> behind increased output instability requires further, more targeted
> readouts to resolve."

---

### 6.4 Connection to Psycholinguistics & Human Cognition (200 words)

**Substantially weakened from the original draft**, which built the entire
parallel on oscillation-as-revision — the claim the confirmatory test does
not support. What remains defensible is narrower and should be framed as
motivation for future work, not as a supported finding.

**What survives**:
- Garden-path stimuli showed the highest false-lead temptation rate of the
  three families (66.7% distractor-in-top5, vs. 25–29% for factual and
  arithmetic; §5.4), consistent with garden-path sentences being
  particularly effective at inducing a temporary wrong reading — this is a
  behavioral fact about the stimuli, not yet a claim about internal revision
- The dissociation gap (H3) is the mechanistically relevant analog worth
  pursuing for a human-comparison study, not oscillation

**What must be dropped or heavily hedged**:
- Any claim that oscillation depth is a model-internal analog of human
  reading-time slowdowns — the mechanism this rested on (candidate
  revision) is not supported
- The "r ≈ ?" placeholder correlation with Natural Stories reading times —
  this analysis was never run and should not be implied as pending
  confirmation of the revision story; if run in future work, it should be
  correlated against the dissociation gap (H3) rather than oscillation
  (H2), since H3 is the effect that is actually robust

**Writing template**:
> "Garden-path stimuli produced the strongest false-lead temptation of the
> three families we tested (66.7% of items had the distractor reach the
> model's final top-5, versus 25–29% for factual and arithmetic prompts),
> consistent with the classical psycholinguistic observation that reduced
> relative clauses are unusually effective at inducing a temporary incorrect
> parse. We had originally hypothesized that model-internal oscillation
> would provide a mechanistic analog to human garden-path reading-time
> slowdowns; our confirmatory test does not support this, and we withhold
> the claim. The dissociation gap — confidence preceding correctness — is
> the more promising candidate for a future correlational study against
> human reading times, since it is the effect that survives targeted
> confirmatory testing in this work."

---

### 6.5 Limitations & Future Work (350 words)

#### Limitations

**1. Small model size**
- Qwen 1.7B: open-weight advantage but smaller than frontier models
- Future: replicate on Qwen 4B (fresh lens fit already scoped) and other
  open-weight families

**2. Modest complete-pair counts**
- Primary H1/H3 rest on 14 complete holdout pairs; confirmatory metrics on
  18–19. Larger than run 1's n=6, but still modest
- Future: expand the matched-pair stimulus set further, particularly
  arithmetic and garden-path, whose behavioral accuracy (70.8%, 75.0%) has
  more room to grow than factual's 95.8%

**3. Single registered split**
- All results reflect one pre-registered 60/40 dev/holdout split (seed 42);
  we did not resample the split itself, only θ and band within it
- Future: report split-resampling stability (e.g., repeated random splits,
  each with fresh θ estimation) as an additional robustness axis

**4. Oscillation mechanism unresolved**
- We know oscillation increases (primary metric, θ-fragile) but not why —
  the confirmatory 2AFC test rules out simple candidate-revision but does
  not identify an alternative mechanism
- Future: extend the confirmatory readout beyond a 2-way comparison (e.g.,
  top-k restricted entropy), or a causal ablation of the distractor
  direction to test whether it drives the churn at all

**5. Single-token targets**
- J-lens vocabulary limitation; multi-token completions (reasoning steps)
  not captured
- Future: extend to multi-token or logit-lens-only readouts

**6. English-dominant, single-language stimuli**
- Future: cross-linguistic replication

**7. Correlational, not causal**
- We observe the dissociation gap and oscillation but don't manipulate what
  causes either
- Future: ablation experiments on the distractor representation specifically

**8. No Natural Stories correlation run**
- Deferred; if run, should target the dissociation gap (H3), not
  oscillation (H2) — see §6.4

#### Future Directions

**Immediate**:
- Qwen 4B replication (fresh lens fit)
- Expand stimuli, especially arithmetic/garden-path, for power
- Logit-lens secondary readout validation (`lens_utils.py`, not yet wired in)

**Medium-term**:
- Causal ablation of the distractor direction, to test the oscillation
  mechanism directly rather than inferring it from a second readout
- Multi-token extension
- Cross-lingual generalization

**Long-term (speculative)**:
- Real-time uncertainty quantification based on the dissociation gap
  specifically (not oscillation, given its unresolved mechanism)
- Pre-answer hallucination detection systems
- Architectural insights for improving model robustness

---

### 6.6 Practical Implications (250 words)

**What to emphasize**: Why the dissociation gap, specifically, matters
beyond academia — this section is now built around H3 rather than H2.

**Applications**:

**1. Hallucination Detection**
- Dissociation gap as a risk marker — the one signature in this study
  robust to every sensitivity check
- Could be monitored at inference time
- No auxiliary models needed (already in lens readouts)

**2. Uncertainty Quantification**
- A model's internal confidence (ΔH collapse) can precede correctness by a
  measurable, false-lead-sensitive margin
- Pre-output signal, unlike temperature scaling or other post-hoc methods

**3. Model Auditing & Safety**
- Inspect "confidently-wrong windows" for patterns across prompt types
- Identify which prompt types (per §5.4, garden-path most of all) are
  highest-risk for false-lead-induced dissociation
- Improve data collection and training with this signal in mind

**4. A Cautionary Methodological Point**
- The oscillation result is itself a useful case study: a single aggregate
  metric suggested a mechanistic story (revision) that a targeted
  confirmatory test did not support. Auditing and interpretability claims
  built on one readout should budget for this kind of check.

**Writing template**:
> "Beyond theoretical interest, the dissociation gap has practical
> implications. It identifies a 'hallucination-prone window' where models
> are most confident yet most vulnerable, and — unlike the oscillation
> signal we initially expected to be the headline result — it survives every
> robustness check we applied. In deployment, monitoring this gap in
> real-time could flag unreliable outputs before they reach users. For model
> auditors, the false-lead temptation rates broken down by family (§5.4)
> offer a starting point for identifying which prompt types carry the
> highest dissociation risk. Just as importantly, the fate of the
> oscillation hypothesis in this study is itself a caution: a plausible,
> appealing single-metric finding did not survive a pre-registered
> confirmatory test, which is exactly the outcome pre-registration and
> dual-readout designs are meant to catch."

---

### 6.7 Concluding Remarks (150 words)

**Key take-home messages**:
1. The confidence–correctness dissociation gap is real, false-lead-sensitive,
   and robust to every analysis choice we tested — the strongest claim this
   paper can make
2. Delayed target-vs-distractor commitment is real in the same robust sense,
   at a narrower readout than originally hypothesized
3. Internal oscillation increases under false lead but is not shown to
   reflect revision between candidate answers; we report it honestly as an
   open question rather than a resolved mechanism
4. Pre-registering a confirmatory readout, and fixing the interpretation
   rule for disagreement in advance, is what let us catch (2) rather than
   report it as a finding

**Final paragraph template**:
> "We began with a simple question: when does a language model know its
> answer? Instrumenting the J-space with layer-wise metrics shows that
> confidence and correctness can measurably dissociate — models are
> sometimes confident well before they are right, and false-lead framing
> widens this window in a way that survives every robustness check we ran.
> A second, more tempting story — that the model's shifting top-1 identity
> reflects active revision between candidate answers — did not survive a
> targeted confirmatory test we pre-registered specifically to check it. We
> report both outcomes, because the discipline of testing an appealing
> mechanistic claim and being willing to walk it back is, in our view, as
> much a contribution here as the dissociation gap itself."

---

## Writing Checklist

- [x] 6.1 Summary (all hypotheses stated, both readouts)
- [x] 6.2.1 Dissociation gap (hallucination angle — headline)
- [x] 6.2.2 Oscillation (reframed: unattributed churn, not revision)
- [x] 6.2.3 Hard controls (reframed: evidence against H2 specificity)
- [x] 6.3 Gurnee et al. context (updated numbers, narrower claims)
- [x] 6.4 Psycholinguistics connection (weakened, redirected toward H3)
- [x] 6.5 Limitations (sensitivity fragility, unresolved H2 mechanism)
- [x] 6.6 Practical implications (H3-centric)
- [x] 6.7 Concluding remarks (honest headline)

---

## Style Notes

- Use "we find" / "our results show" (active voice)
- Avoid overclaiming ("reveals," "proves" → "suggests," "indicates")
- Lead with H3 (dissociation gap), not H2 (oscillation) — this is the
  single biggest change from the original outline
- When mentioning oscillation, always pair the primary result with the
  confirmatory null and the hard-control comparison; never state the
  primary result alone
- Link back to Gurnee et al. throughout
- Acknowledge limitations upfront
- End on future-work note (not closure)

---

## Length Target

- 6.1: 200 words
- 6.2: 450 words (3 subsections)
- 6.3: 300 words
- 6.4: 200 words (shortened — the parallel is weaker now)
- 6.5: 350 words
- 6.6: 250 words
- 6.7: 150 words
- **Total: ~1900 words**

---

## Integration with GPU Results

Qwen3-1.7B run 2 integrated (2026-07-22):
1. ✅ H1–H3 numbers (primary + confirmatory) filled in §6.1 and
   `RESULTS_TEMPLATE.md`
2. ✅ Sensitivity results integrated throughout (`SENSITIVITY_REPORT.md`)
3. ✅ Hard-control comparison integrated (§6.2.3)
4. ⏸ Natural Stories correlation still pending — retarget at H3 if run
5. ⏸ Qwen 4B deferred
6. ✅ Figure paths: `out/figures/H1_*.png`, `H2_*.png`, `H3_*.png`,
   `entropy_curves.png`, `heatmaps/`
7. Remaining work: expand the writing templates above into final
   submission-ready prose (they are structured but still template-shaped)
