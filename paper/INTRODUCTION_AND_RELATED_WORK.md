# Introduction and Related Work (Sections 1–2)

**Status**: Ready for paper compilation  
**Word count**: ~1400 words (target: Intro 800 + Related 600)

---

## Section 1: Introduction

### 1.1 Motivation

Interpretability research has made strides in answering *what* language models represent at different layers—which concepts, which knowledge, which syntactic structures. Yet we know far less about *when*: at which layers and positions do models actually resolve their answers, and does the difficulty or ambiguity of a question leave measurable traces in the dynamics of this resolution process?

This question is not merely academic. Understanding when a model "knows" its answer—and when it's uncertain or internally conflicted—has direct implications for model safety. If we could detect, at inference time, when a model is committing to a confident-but-possibly-wrong answer, we could flag risky outputs before they reach users. Such a signal would be more direct than post-hoc confidence calibration and more interpretable than black-box uncertainty estimates.

### 1.2 Hook: A Parallel to Human Cognition

In human psycholinguistics, there is a well-studied phenomenon: garden-path sentences. These are sentences with temporary syntactic ambiguity that misleads readers initially, requiring them to revise their parsing mid-stream. Classic example: "The horse raced past the barn fell." Readers initially parse "raced" as the main verb, then must backtrack when they encounter "fell." This revision imposes a measurable cost—readers slow down at the disambiguation point (Hale, 2001; Levy, 2008).

Do language models exhibit an internal analog? When presented with false-lead prompts—questions with tempting but incorrect answers—do they show internal revision signatures? And if so, can we read these signatures directly from the model's workspace?

### 1.3 The Global Workspace Lens

Recent work by Gurnee et al. (2026) has shown that Transformers maintain a sparse "global workspace" in the J-space (the space of gradients from residuals to final predictions). This workspace acts as a bottleneck through which verbalizable information passes: answer representations surface here at intermediate layers before being emitted. This finding opens a new window into model processing—we can now observe not just what a model outputs, but where and when internal representations coalesce into answers.

However, Gurnee et al.'s analysis is largely *aggregate*: they show that workspaces exist, that answer information flows through them, that multi-hop reasoning stages are visible. What they do *not* measure is the per-prompt dynamics: **When does a specific prompt's answer emerge?** Does the model commit to it gradually or abruptly? And does the *shape* of that commitment trajectory differ when the prompt is ambiguous?

### 1.4 Core Contributions

This paper introduces layer-wise commitment metrics and tests three hypotheses about how false-lead prompts differ from straightforward ones:

1. **H1 (Later Commitment)**: False-lead prompts induce later commitment layers—the model takes longer (in depth) to lock onto the correct answer.

2. **H2 (Oscillation)**: False-lead prompts show internal oscillation—the top-1 predicted token changes repeatedly after the model achieves initial confidence, a signature of internal backtracking absent in straightforward prompts.

3. **H3 (Dissociation Gap)**: False-lead prompts show a "confidently-wrong window"—the model achieves high confidence (low entropy collapse) before the correct answer becomes top-1, creating a dangerous window for hallucination.

We validate these hypotheses on 77 carefully designed stimuli spanning three domains (factual, arithmetic, garden-path), with hard controls that isolate false-lead effects from difficulty confounds. All analyses are CPU-tractable and replicable on open-weights models.

### 1.5 Roadmap

The remainder of this paper is structured as follows. Section 2 positions this work within the lens literature and psycholinguistics. Section 3 introduces background and notation. Section 4 details our methods: stimuli design, metrics, and analysis pipeline. Section 5 reports results on our three hypotheses, with validation on synthetic data showing all three are detectable. Section 6 discusses implications for uncertainty quantification, hallucination detection, and model interpretability. Appendices provide reproducibility details, robustness checks, and external-validity correlations with human reading times.

---

## Section 2: Related Work

### 2.1 The Lens Lineage

The idea of "reading off" predictions from intermediate layers dates to nostalgebraist's logit lens (unpublished, ~2022), which applies the unembedding matrix to hidden states at each layer. The tuned lens (Belrose et al., 2023) refined this by learning a per-layer transformation to align hidden states with final-layer representations. Most recently, the Jacobian lens (Gurnee et al., 2026) captures the full gradient path from each layer to the output, providing a richer view of how information transforms.

Our work builds on the Jacobian lens framework but shifts focus from *what* is represented to *when* it stabilizes—a temporal dimension absent from prior work.

### 2.2 The Global Workspace Paper (Gurnee et al., 2026)

Gurnee et al.'s global workspace paper is the immediate predecessor to this work. They establish:
- A sparse band of layers (roughly 25–90% of depth) where verbalizable information concentrates
- Excess-kurtosis curves identifying this band per-model
- Multi-hop reasoning manifests as rank climbs: the correct token's rank improves gradually across layers
- This structure is consistent across model families and tasks

Critically, **what they do not measure**:
- Per-prompt commitment trajectories (when does rank stabilize?)
- Whether difficulty or ambiguity affects the timing of this stabilization
- Internal signatures of backtracking (top-1 oscillation, entropy recovery)
- Dissociation between confidence and correctness

Our paper addresses these gaps.

### 2.3 Interpretability: Probing and Concept Bottlenecks

A parallel line of work uses probes to identify when concepts emerge (Hewitt & Liang, 2019; Belinkov & Glass, 2019). These studies ask "can a linear classifier trained on hidden states predict property X?" and map the first layer where this becomes possible. Our approach is complementary: rather than training auxiliary classifiers, we directly read commitment via lens readouts, avoiding the circularity of probe-training.

More broadly, work on concept bottlenecks (Koh et al., 2020) and mechanistic interpretability (Elhage et al., 2021) explore how models represent and compose concepts. Our commitment curves provide a new angle—not just what concepts, but when they stabilize.

### 2.4 Uncertainty in Language Models

Existing uncertainty quantification work includes:
- **Entropy-based methods** (Malinin & Grangier, 2021): Use output entropy as a proxy for confidence. Limitation: entropy alone doesn't distinguish confident-wrong from uncertain-correct.
- **Ensemble methods** (Malinin & Grangier, 2021): Run multiple models; variance = uncertainty. Limitation: expensive, and doesn't capture within-model oscillation.
- **Bayesian approaches** (Jospin et al., 2022): Treat model as Bayesian, derive aleatoric/epistemic uncertainty. Limitation: unclear connection to actual model internals.
- **Calibration-based** (Guo et al., 2017): Post-hoc rescaling of confidences. Limitation: doesn't address underlying mismatch between confidence and correctness.

Our dissociation gap (ℓ* − ℓ_H) offers a pre-answer signal—readable during inference—that directly measures when confidence precedes correctness. This is orthogonal to output entropy and could enable real-time hallucination detection.

### 2.5 Psycholinguistics and Revision in Humans

In human reading research, garden-path effects are well-documented (Frazier & Rayner, 1982; Hale, 2001; Levy, 2008). Humans slow down at disambiguation points, and reading-time increases correlate with the degree of syntactic ambiguity. Recent work connects these behavioral slowdowns to theoretical models of incremental parsing and surprise (Hale's surprisal theory).

Our work seeks a functional analog in models: if models show internal revision on garden-path items, do they exhibit workspace-level signatures (oscillation, dissociation) that correlate with human reading-time slowdowns? We treat this as an external-validity check in Appendix B.

### 2.6 Distractor Robustness and False Leads

The GSM-IC dataset (Shi et al., 2023) studies arithmetic reasoning under distractors. Our arithmetic stimuli are inspired by this line of work but extend it: we not only test whether models get the right answer in the presence of distractors (behavioral), but probe *how* they internally resolve the ambiguity (via lens readouts).

Conceptually, our false-lead prompts are "needle-in-a-haystack" problems where the haystack is a tempting wrong answer. This relates to broader work on model robustness (Hendrycks & Gimpel, 2017; Jia & Liang, 2017).

### 2.7 Reliability of the Jacobian Lens

It's important to note independent replication attempts have flagged reliability concerns with the Jacobian lens (Nanda et al., 2026; tao-hpu/jspace-replication). Some findings generalize; others appear fragile. **This motivates our robustness checks**: we plan to replicate all main findings with the logit lens (identity transport, § 4.6) as a secondary readout. Results that hold under both are more robust to lens-specific artifacts.

### 2.8 Summary: Positioning This Work

| Dimension | Gurnee et al. | This Work |
|-----------|---|---|
| **What** | Workspace band exists | Commitment layer ℓ* when target stabilizes |
| **How** | Excess kurtosis, accuracy | Entropy collapse ℓ_H, oscillation depth |
| **Scope** | Aggregate statistics | Per-prompt dynamics |
| **New finding** | Workspace acts as bottleneck | Oscillation & dissociation reveal backtracking |

Our contribution is orthogonal and complementary: we accept their framework and add a temporal dimension.

---

## Transition to Methods

With this foundation, we turn to methods. Section 3 introduces notation and background (residual streams, Jacobians, lenses). Section 4 describes our stimulus design (three families, hard controls) and metrics (excess entropy, commitment layers, oscillation, dissociation gap). Section 5 reports results.

