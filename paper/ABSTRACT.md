# Abstract

**Title**: Commitment Dynamics in the J-Space: When Do Language Models Commit to Answers, and What Happens Under False-Lead Temptation?

---

Language models must commit to answers progressively through their network layers. But **when** do they commit, and **what happens when plausible wrong answers are initially tempting?** We study answer commitment in a mechanistic residual-stream readout—the Jacobian lens—which maps each layer's activations directly into predicted logits without running the forward pass through upper layers.

On 163 carefully designed stimuli spanning factual knowledge (51 items), arithmetic reasoning (56 items), and garden-path sentences (56 items), we test three hypotheses about false-lead effects—when a prompt initially suggests an incorrect answer before revealing the correct one:

1. **H1 (Delayed Commitment)**: The layer where models lock in the target answer (ℓ*) occurs **later** under false-lead than straightforward conditions.

2. **H2 (Internal Oscillation)**: The top-1 predicted token **oscillates** (changes identity) in middle layers under false-lead, indicating real-time backtracking and revision.

3. **H3 (Dissociation Gap)**: A critical **confidence-correctness mismatch** emerges: models become confident (entropy collapses at layer ℓ_H) but are still wrong (target hasn't ranked top-1 yet). The gap ℓ* − ℓ_H measures how long models "confidently hallucinate" before correcting course.

We propose that false-lead effects in LLMs reflect a **bottleneck phenomenon** (global workspace theory): answer representations progressively emerge and broadcast through a narrow channel, causing predictable delays and oscillations when evidence conflicts. We apply the Jacobian lens to Qwen3-1.7B on 163 stimuli (lens fitted on 1000×128 FineWeb prompts) and evaluate pre-registered holdout tests.

**Real-model results.** H2 is strongly supported: false-lead prompts increase oscillation depth (median 3.0 vs 1.0; paired median Δ=+2.0; Wilcoxon p=4.89×10⁻⁵, n=32). H1 and H3 move in the predicted direction (later commitment: median ℓ* 22 vs 20, Δ=+2.0; larger gap: median 4.5 vs 0.0, Δ=+4.0) but are not significant at α=0.05 with the current complete-pair counts (both p=0.0625, n=6). False-lead items also show longer distractor-lead stretches (median 11.0 vs 2.5 layers), validating temptation in the lens readout.

This framework unifies mechanistic interpretability (where do answers emerge?) with cognitive science (how do minds revise initial commitments?). The clearest real-model signature is **internal oscillation under false lead**—a revision signal visible in Jacobian-lens readouts before the final answer—while delayed commitment and dissociation-gap claims remain provisional pending higher-powered complete-pair analyses and larger models.

---

## Keywords

Jacobian lens, commitment layer, global workspace, false-lead effects, garden-path sentences, interpretability, residual-stream readout, language model reasoning

