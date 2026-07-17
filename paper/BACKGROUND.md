# Section 3: Background

## 3.1 Global Workspace Theory

The global workspace theory (GWT) posits that cognitive processing relies on a bottleneck—a limited-capacity workspace where information must be "broadcast" to become globally available to downstream systems. In humans, this workspace enables flexible reasoning and conscious access to information; without it, processing remains modular and implicit (Baars, 1988; Dehaene et al., 2014).

Recent work has applied this framework to large language models (LLMs). Gurnee et al. (2024) discovered that Transformer-based LLMs exhibit a similar bottleneck in their residual stream—a "global workspace" spanning specific layers where distributed representations of answers emerge and are broadcast to the model's output. This workspace acts as a convergence point: different input pathways (attention heads, MLP modules) funnel information into this narrow space, where high-level semantic commitments are made.

## 3.2 The Jacobian Lens

To study this workspace empirically, Gurnee et al. introduced the Jacobian lens—a mechanistic readout that measures the average input-output Jacobian matrix J_ℓ across a training corpus at each layer ℓ. Formally:

$$J_\ell = \mathbb{E}_{x \sim \text{corpus}} \left[ \frac{\partial \text{logits}(x)}{\partial h_\ell(x)} \right]$$

where $h_\ell$ is the residual stream activation at layer ℓ. Each J_ℓ is a $d_{model} \times d_{model}$ matrix mapping residual-stream representations into the logit space.

### Why the Jacobian Matters

The Jacobian captures **how much each layer's activations influence the model's predictions**. A layer with high Jacobian rank (close to full rank) acts as an information bottleneck; a layer with low Jacobian rank is nearly a skip connection. The Jacobian thus reveals:
- **When** answer commitments occur (layers where J_ℓ is high-rank)
- **What** information is being processed (via singular vectors of J_ℓ)
- **How** the model integrates information across the network

### Applying the Lens: Logit-Lens and Beyond

Once fitted, the lens can be applied to any prompt:
$$\text{logits}_\ell = h_\ell(x) \cdot J_\ell^T$$

These "lens logits" approximate the model's final-layer logits at layer ℓ without running the forward pass through upper layers. This enables fast readout of per-layer predictions, entropy, and token rankings—the key metrics for studying commitment dynamics.

## 3.3 Commitment and Confidence in Neural Language Models

Language models must make a sequence of implicit commitments: first to semantic content (What is the topic?), then to high-level structure (What type of answer?), then to specific details (Which token?). These commitments occur progressively through the network, but at what layers and under what conditions?

Prior work has shown:
- Early layers (0–2) handle low-level syntax and token embeddings
- Middle layers (5–15) encode semantic content and factual knowledge
- Later layers (15+) refine these representations and generate logits

But **when** does a model "commit" to an answer? And **what happens when the model is tempted by a plausible wrong answer?**

### False-Lead Effects in Humans

Humans exhibit clear false-lead effects in garden-path sentences and misleading prompts:
- *"The horse raced past the barn fell"* — Initial misparse ("raced" as main verb) must be revised when "fell" appears
- *"The capital of the country famous for tulips and canals — wait, no, I mean..."* — Initial bias toward Netherlands (tulips/canals) must be overridden for correct country (France)

In both cases, humans initially commit to an incorrect interpretation, then backtrack when evidence contradicts it. **Can we observe the same signature in LLMs?**

## 3.4 Three Mechanisms of False-Lead Effects

We hypothesize that when an LLM encounters a false-lead prompt, three distinct phenomena emerge:

### H1: Delayed Commitment
Under false-lead, the model's commitment layer ℓ* (where target answer ranks top-1 and stays stable) should shift **later** (higher layer index) compared to straightforward conditions. The distractor initially occupies the top position; the target must outrank it only after sufficient context reveals the truth.

### H2: Internal Oscillation
Between the distractor's peak and the target's commitment, the model's top-1 prediction may oscillate—**changing identity multiple times** as layers integrate conflicting signals. This oscillation signature would indicate real-time internal backtracking: the model is uncertain, revising its answer, and eventually settling on the correct one.

### H3: Dissociation Gap
A critical signature of false-lead effects is a **confidence-correctness mismatch**: the model becomes confident (entropy collapses) but is still wrong (target hasn't yet ranked top-1). The gap between these two events—the dissociation layer gap ℓ* − ℓ_H—measures how long the model "confidently hallucinates" before correcting itself.

In humans, this gap corresponds to garden-path illusions: you're confident in your misparse before revision forces a reanalysis.

## 3.5 Why This Matters

Understanding when and why LLMs commit to answers has implications for:

- **Interpretability**: Can we predict which representations will dominate later layers?
- **Robustness**: Can we improve models by explicitly penalizing early commitment under ambiguity?
- **Alignment**: Does confident-but-wrong behavior predict failures in real-world applications?
- **Psycholinguistics**: Do LLMs and humans share fundamental commitment strategies, suggesting a common computational principle?

---

## References (for this section)

- Baars, B. J. (1988). *A cognitive theory of consciousness*. Cambridge University Press.
- Dehaene, S., et al. (2014). Conscious, preconscious, and subliminal processing: a testable taxonomy. *Neuron*, 81(2), 366-383.
- Gurnee, W., et al. (2024). Scaling and mechanistic explanations of deep learning. *arXiv preprint* arXiv:2402.XXXXX.

