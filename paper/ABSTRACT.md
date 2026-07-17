# Abstract

**Title**: Commitment Dynamics in the J-Space: When Do Language Models Commit to Answers, and What Happens Under False-Lead Temptation?

---

Language models must commit to answers progressively through their network layers. But **when** do they commit, and **what happens when plausible wrong answers are initially tempting?** We study answer commitment in a mechanistic residual-stream readout—the Jacobian lens—which maps each layer's activations directly into predicted logits without running the forward pass through upper layers.

On 163 carefully designed stimuli spanning factual knowledge (51 items), arithmetic reasoning (56 items), and garden-path sentences (56 items), we test three hypotheses about false-lead effects—when a prompt initially suggests an incorrect answer before revealing the correct one:

1. **H1 (Delayed Commitment)**: The layer where models lock in the target answer (ℓ*) occurs **later** under false-lead than straightforward conditions.

2. **H2 (Internal Oscillation)**: The top-1 predicted token **oscillates** (changes identity) in middle layers under false-lead, indicating real-time backtracking and revision.

3. **H3 (Dissociation Gap)**: A critical **confidence-correctness mismatch** emerges: models become confident (entropy collapses at layer ℓ_H) but are still wrong (target hasn't ranked top-1 yet). The gap ℓ* − ℓ_H measures how long models "confidently hallucinate" before correcting course.

We propose that false-lead effects in LLMs reflect a **bottleneck phenomenon** (global workspace theory): answer representations progressively emerge and broadcast through a narrow channel, causing predictable delays and oscillations when evidence conflicts. To test this, we apply the Jacobian lens to Qwen3-1.7B and systematically measure commitment dynamics across our 163 stimuli. We validate our methodology on synthetic traces and find that our analysis pipeline successfully detects all three hypothesized patterns (H1, H2, H3) with strong signal (p < 0.001 in simulation).

This framework unifies mechanistic interpretability (where do answers emerge?) with cognitive science (how do minds revise initial commitments?). If supported by real model data, these findings would reveal oscillation as a novel mechanistic signature of internal uncertainty resolution not captured by standard logit-lens approaches, and would parallel human psycholinguistics: garden-path sentence effects show similar commitment delays and confidence-correctness mismatches in reading times and neural responses.

---

## Keywords

Jacobian lens, commitment layer, global workspace, false-lead effects, garden-path sentences, interpretability, residual-stream readout, language model reasoning

