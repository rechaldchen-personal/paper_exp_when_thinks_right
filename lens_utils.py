"""
lens_utils.py — Utilities for J-lens and logit-lens readouts.

The paper reports results from two independent lens methods to rule out J-lens
false positives. This module provides both transport paths.
"""

import torch
from typing import Dict, Tuple


def apply_logit_lens(
    model,
    input_ids: torch.Tensor,
    positions: list = None,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Logit lens (identity transport): residual -> lm_head directly.

    This is the "naive" identity-based lens: h_ℓ → norm → lm_head, ignoring
    the Jacobian. It's a sanity check and robustness baseline.

    Args:
        model: HuggingFace model (already loaded and in eval mode)
        input_ids: (batch=1, seq_len) token ids
        positions: list of positions to read at; if None, all positions

    Returns:
        logit_lens_readouts: dict {layer_idx: tensor[n_positions_requested, vocab]}
        final_logits: (seq_len, vocab) from model's final output

    Note: Only works for models with .model.layers and standard norm + lm_head.
    """
    if positions is None:
        positions = list(range(input_ids.shape[1]))

    logits_by_layer = {}
    device = next(model.parameters()).device

    # Hook to capture residuals
    residuals = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            residuals[layer_idx] = h.detach()  # (batch, seq, d_model)
        return hook

    blocks = model.model.layers
    hooks = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]

    with torch.no_grad():
        output = model(input_ids)
        final_logits = output.logits[0]  # (seq_len, vocab)

    for hook in hooks:
        hook.remove()

    # Apply identity transport: residual -> norm -> lm_head
    norm_layer = model.model.norm
    lm_head = model.lm_head
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

    with torch.no_grad():
        for layer_idx in sorted(residuals.keys()):
            h_layer = residuals[layer_idx][0]  # (seq, d_model), remove batch dim
            logits_at_layer = []

            for pos in positions:
                h_pos = h_layer[pos : pos + 1]  # (1, d_model)
                h_normed = norm_layer(h_pos)
                logits_pos = lm_head(h_normed)  # (1, vocab)
                logits_at_layer.append(logits_pos[0])  # (vocab,)

            logits_by_layer[layer_idx] = torch.stack(logits_at_layer)  # (n_pos, vocab)

    return logits_by_layer, final_logits


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of softmax(logits). logits: (..., vocab)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """0-based rank of token_id in descending logits. logits: (vocab,)."""
    return int((logits > logits[token_id]).sum().item())
