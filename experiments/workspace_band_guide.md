# Workspace Band Identification Guide

Per Gurnee et al. (2026), the "workspace" is a sparse band of layers where verbalizable information surfaces. Early and late layers don't reliably represent answers.

The paper uses **hardcoded band [0.25, 0.90]** as a default. For proper results, **identify the band per-model** using the criteria below.

## Criteria (from Gurnee et al. Fig 28)

Three complementary signals identify the band:

### 1. Excess Kurtosis

Excess kurtosis of lens readout entropy: peaks within the workspace.

```python
# Pseudo-code; adapt to your data
kurtosis_by_layer = []
for layer_idx in range(n_layers):
    ents = [entropy[layer_idx][t] for t in range(n_positions)]
    k = scipy.stats.kurtosis(ents)
    kurtosis_by_layer.append(k)

band_start = argmax(kurtosis_by_layer[:middle])
band_end = argmax(kurtosis_by_layer[middle:]) + middle
```

### 2. Autocorrelation of Rank Trajectories

Within the workspace, target rank stays stable; outside, it's noisy.

```python
# For each prompt, compute autocorr of target_rank[layer][query_pos]
# High autocorr = workspace; low autocorr = noise
autocorr_by_layer = []
for layer_idx in range(n_layers):
    ranks = [target_rank[layer_idx][q] for q in query_positions]
    ac = np.correlate(ranks, ranks, mode='same')[len(ranks)//2:]
    autocorr_by_layer.append(ac[1])  # lag-1 autocorr
```

### 3. Next-Token Accuracy on Held-Out Examples

Models' top-1 predictions on held-out text: peaks in the workspace.

```python
# Run lens on a small corpus of held-out examples
# Compute fraction of correct next-token predictions per layer
correct_by_layer = []
for layer_idx in range(n_layers):
    preds = argmax(logits[layer_idx]) for each position
    acc = mean(preds == true_tokens)
    correct_by_layer.append(acc)
```

## How to Run

**Once you have GPU access:**

```bash
# 1. Generate a small corpus for band identification
#    (100–500 examples from a generic corpus like FineWeb)

# 2. Run lens on this corpus to collect entropy, ranks, logits

# 3. Compute kurtosis, autocorr, accuracy curves (tools TBD)

# 4. Visually inspect the three signals; band [start, end] should
#    correspond to the "peak" region in all three

# 5. Update 02_run_experiment.py and 03_analyze.py:
#    --band 0.15 0.92  (or whatever you found)
```

## Default Fallback

If you can't identify the band, the hardcoded [0.25, 0.90] works for most Qwen models but may miss layer-specific effects.

## Output for Paper

In Appendix C, report:
- Kurtosis curve (by layer)
- Autocorrelation curve (by layer)
- Next-token accuracy curve (by layer)
- The band you selected + justification
