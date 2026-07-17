# Natural Stories Integration — External Validity Check (Section 7.4)

The paper adapts garden-path sentences from the Natural Stories corpus (Futrell et al., 2018), which has accompanying human reading times. This allows us to correlate internal oscillation/dissociation metrics with external behavioral evidence.

## Dataset

**Natural Stories**: 10 stories (~100 sentences each) from published literature, annotated with:
- Per-sentence human reading times (self-paced reading)
- Syntactic structure annotations
- Garden-path status

**Available**: https://huggingface.co/datasets/coryshain/natural_stories

## Subset Selection for Paper

### Garden-Path Items in Natural Stories
Examples of sentences that induce temporary ambiguity:
- "The horse raced past the barn fell" (main/reduced relative clause ambiguity)
- "The cat chased by the dog into the yard hid" (same structure)
- "The flower arranged by the gardener in the vase bloomed"

**Strategy**: 
1. Curate 5–10 garden-path sentences from Natural Stories
2. Create both versions:
   - **False-lead**: Original ambiguous sentence
   - **Straightforward**: Disambiguated version (add "that was")
3. Measure both:
   - Internal: oscillation_depth, l_star, dissociation_gap (from J-lens)
   - External: human reading-time slowdown at disambiguation region

### Disambiguation Region
The point where humans show increased reading time (spillover effect):
- In "The horse raced past the barn **fell**": the word "fell" disambiguates
- Human RT: ~50–100ms slowdown vs. unambiguous baseline
- LLM oscillation: should correlate with this region

## Expected Correlation

**Hypothesis 7.4**: Models with higher internal oscillation/dissociation should show a stronger analog of human garden-path effects.

Predictions:
- **Oscillation depth** (# top-1 identity changes after commitment) correlates with human reading-time slowdown (r > 0.4, p < 0.05)
- **Dissociation gap** (l_star - l_H) correlates with RT slowdown
- **Models that show internal backtracking** (in J-lens) are more human-like in this regard

## Implementation (GPU phase)

```bash
# 1. Load Natural Stories dataset
python -c "from datasets import load_dataset; ds = load_dataset('coryshain/natural_stories')"

# 2. Curate ambiguous + unambiguous pairs (manual + automated)

# 3. Run 02_run_experiment.py on the Natural Stories subset

# 4. Extract human reading times for disambiguation region

# 5. Compute correlation:
#    correlate(oscillation_depth, human_rt_slowdown)
#    correlate(dissociation_gap, human_rt_slowdown)

# 6. Report in Results + visualize as scatter plots
```

## Output for Paper

**Figure in Results section**:
- X-axis: LLM oscillation depth or dissociation gap
- Y-axis: Human reading-time slowdown (ms)
- Points: individual Natural Stories sentences
- Trend line + R-squared + p-value

**Table in Results**:
| Metric | Correlation (r) | p-value | n_items |
|--------|-----------------|---------|---------|
| oscillation_depth | ? | ? | ~8 |
| dissociation_gap | ? | ? | ~8 |
| l_star | ? | ? | ~8 |

## Notes

- This is **purely correlational** (not causal), but provides external validity
- Limits: Small sample of Natural Stories items; English-specific
- Future work: Extend to other languages, other garden-path corpora

## References

Futrell, R., Gibson, E., Tily, H., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, E. (2018). The Natural Stories Corpus. arXiv preprint arXiv:1805.04680.
