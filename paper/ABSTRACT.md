# Abstract

**Title**: Commitment Dynamics in the J-Space: When Do Language Models Commit to Answers, and What Happens Under False-Lead Temptation?

---

Language models must commit to answers progressively through their network layers. But **when** do they commit, and **what happens when plausible wrong answers are initially tempting?** We study answer commitment in a mechanistic residual-stream readout—the Jacobian lens—which maps each layer's activations directly into predicted logits without running the forward pass through upper layers.

On 156 stimuli spanning factual knowledge, arithmetic reasoning, and garden-path sentences (72 strictly matched straightforward/false-lead pairs, 24 per family, plus 12 hard controls with no tempting distractor), we pre-registered three hypotheses about false-lead effects—when a prompt initially suggests an incorrect answer before revealing the correct one:

1. **H1 (Delayed Commitment)**: The layer where models lock in the target answer (ℓ*) occurs **later** under false-lead than straightforward conditions.

2. **H2 (Internal Oscillation)**: The top-1 predicted token **oscillates** (changes identity) in middle layers under false-lead, indicating real-time backtracking and revision.

3. **H3 (Dissociation Gap)**: A critical **confidence-correctness mismatch** emerges: models become confident (entropy collapses at layer ℓ_H) but are still wrong (target hasn't ranked top-1 yet). The gap ℓ* − ℓ_H measures how long models "confidently hallucinate" before correcting course.

For each hypothesis we report two pre-registered readouts: a **primary** metric over the full vocabulary (as originally specified) and a **confirmatory** metric restricted to {target, distractor}, added because the primary oscillation metric cannot distinguish wavering between the two candidate answers from churn among unrelated tokens—only the former would support a "revision" interpretation. The interpretation rule for every combination of outcomes was fixed before this run (`PRE_REGISTRATION_AMENDMENT.md`), and both readouts are reported regardless of which supports the hypothesis. We apply the Jacobian lens to Qwen3-1.7B (lens fitted on 1000×128 FineWeb prompts) and evaluate one-tailed exact signed-rank tests on a pre-registered 60/40 dev/holdout split, gated by a per-family behavioral pre-screen (all three families cleared 70%+ straightforward top-1 accuracy).

**Results.** H3 is the best-supported and most robust finding: the dissociation gap is larger under false-lead in both the primary (median Δ=+1.0 layer, p=0.011, n=14 pairs) and confirmatory readout (median Δ=+4.0 layers, p=0.0016, n=18), and the confirmatory version remains significant across every pre-registered sensitivity setting (θ at the 70th/80th/90th percentile; band [0.20,0.95] through [0.30,0.85]), while the primary version is significant only near the pre-registered default. **H2 does not support a revision interpretation.** The primary oscillation metric is significant (median Δ=+0.5, p=0.0077, n=28), but the confirmatory target-vs-distractor version is not (p=0.46), and hard-control items—which contain no tempting distractor—oscillate about as much as false-lead items (median 2.0 vs. 2.5, both above straightforward's 1.0). Per our pre-specified rule, this pattern is reported as increased top-1 churn of unclear origin, not as evidence of the model wavering between candidate answers. **H1** is null under the primary metric (p=0.109) but significant under the confirmatory one (median Δ=+2.0, p=0.0021, robust across all sensitivity settings): false-lead prompts delay the point where the target overtakes the distractor specifically, without necessarily delaying commitment against the full vocabulary.

This framework unifies mechanistic interpretability (where do answers emerge?) with cognitive science (how do minds revise initial commitments?), though the present data support a narrower claim than that framing originally suggested. The clearest and most robust real-model signature is the **confidence–correctness dissociation** (H3): false-lead prompts create a window in which the model is confident before it is correct, and this window is measurable and stable across analysis choices. Delayed target-vs-distractor commitment (H1) is supported in the same robust sense. Oscillation (H2) is real but its mechanism—candidate revision versus general instability—remains open, and we report it as such rather than as a resolved "internal backtracking" signature.

---

## Keywords

Jacobian lens, commitment layer, global workspace, false-lead effects, dissociation gap, garden-path sentences, interpretability, residual-stream readout, language model reasoning
