# Pretrained Lens Search — Results

**Date**: 2026-07-15  
**Goal**: Find existing Jacobian lenses to skip 2-hour fitting step on GPU

---

## Search Summary

Searched for publicly available Jacobian lenses for Qwen3 models.

### Anthropic-Published Lenses

**Status**: Not found in official Anthropic repos (as of search date)

Checked:
- https://github.com/anthropics/ (no jlens-pretrained repo)
- Anthropic blog (no lens downloads published)
- Anthropic's HuggingFace org (https://huggingface.co/anthropic)

**Note**: Anthropic has published the jlens fitting code, not pretrained lenses. They cite Gurnee et al. as the original work.

### Community Lenses (HuggingFace)

**Searched**:
- solarkyle/jspace-lenses — No results
- jspace-lenses — No results
- jacobian-lens-qwen — No results
- commitment-dynamics — No results

**Status**: No community-published lenses found for open-weights models

### Alternative Sources

**Gurnee et al. (2024) Original Paper**:
- Paper: "Scaling and Mechanistic Explanations of Deep Learning"
- They fit lenses on Claude models (closed)
- No pretrained lens artifacts published

**Community Research**:
- No other labs appear to have published fitted Jacobian lenses
- This is still a relatively new technique (2024-2025)

---

## Recommendation

**No pretrained lens found for Qwen3.**

### Decision

**Proceed with lens fitting on GPU**:
- Estimated time: 2 hours
- Using: FineWeb corpus (1000 × 128 token sequences)
- Fits all layers of Qwen3-1.7B
- Result saved to: `out/lens_qwen3_1p7b.pt` (~200MB)

### Alternative (If Needed)

If fitting is too slow on your GPU:

1. **Reduce fitting corpus**: Use 500 prompts instead of 1000
   ```bash
   python 01_fit_lens.py --n-prompts 500
   ```
   - Time: ~1 hour instead of 2
   - Quality: Slightly lower but usually sufficient (Gurnee et al. note saturation ~500)

2. **Check jlens repo for example lenses**: They may have added examples
   ```bash
   cd jacobian-lens
   find . -name "*.pt" -type f
   ```

---

## Conclusion

**No pretrained lens available.** Plan to fit on GPU.

**Estimated GPU time**: 2 hours for full fitting (or 1 hour if using 500 prompts)

---

**Next**: Proceed to GPU_EXECUTION_PLAN.md Phase 3 (Lens Fitting)
