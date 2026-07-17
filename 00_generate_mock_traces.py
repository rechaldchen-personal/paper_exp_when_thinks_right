#!/usr/bin/env python
"""
00_generate_mock_traces.py — Generate synthetic traces for testing the analysis
pipeline (03_analyze.py) without GPU access.

Usage:
    python 00_generate_mock_traces.py --out out/traces_mock.json

Generates realistic per-stimulus trace data matching 02_run_experiment.py's output
structure. Entropy/rank patterns designed to exhibit the hypothesized signatures:
  - false_lead items: later commitment layer, oscillation, dissociation gap
  - straightforward items: early commitment, stable entropy collapse
  - hard_control items: high entropy but no false target
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import random
import math


def generate_layer_sequence(
    n_layers: int,
    n_positions: int,
    condition: str,
    seed: int,
    vocab_size: int = 50000,
) -> Dict[str, Any]:
    """Generate synthetic entropy and rank traces for one stimulus.

    Args:
        n_layers: number of layers (e.g., 25)
        n_positions: sequence length in tokens
        condition: "straightforward", "false_lead", or "hard_control"
        seed: random seed for reproducibility
        vocab_size: vocabulary size (for realistic rank/logp)

    Returns:
        dict with keys: entropy, baseline_entropy, target_rank, distractor_rank,
                        target_logp, distractor_logp, top1_id, layers, n_positions
    """
    random.seed(seed)

    layers = list(range(0, n_layers))
    query_pos = n_positions - 1

    # Baseline entropy: higher in early layers, stable in later
    baseline_ents = [3.5 + 0.8 * math.exp(-i / 5) for i in range(n_layers)]

    # Entropy and ranks: shape depends on condition
    entropy = []
    target_rank = []
    distractor_rank = []
    target_logp = []
    distractor_logp = []
    top1_id = []

    if condition == "straightforward":
        # Early, stable commitment to target
        for layer_idx in range(n_layers):
            pos_data = []
            t_rank_data = []
            d_rank_data = []
            t_logp_data = []
            d_logp_data = []
            top1_data = []

            for pos in range(n_positions):
                # Progressive entropy reduction toward query position
                progress = pos / n_positions
                layer_progress = layer_idx / n_layers
                # Entropy drops sharply around 60% of depth
                commit_depth = 0.6
                if layer_progress > commit_depth:
                    pos_ent = baseline_ents[layer_idx] * (0.3 + 0.7 * (1 - layer_progress))
                else:
                    pos_ent = baseline_ents[layer_idx] * (0.8 + 0.2 * layer_progress)

                pos_data.append(pos_ent)

                # Target quickly becomes top-1 and stays there
                if layer_progress > 0.55:
                    t_r = 0
                    d_r = max(1, int(vocab_size * 0.1 * random.random()))
                else:
                    t_r = max(0, int(vocab_size * 0.2 * (1 - layer_progress)))
                    d_r = max(0, int(vocab_size * 0.15 * (1 - layer_progress)))

                t_rank_data.append(t_r)
                d_rank_data.append(d_r)

                # Log-probs: target strong after commitment
                t_lp = -0.5 - 5 * (t_r / vocab_size) if t_r < vocab_size else -10
                d_lp = -2.0 - 5 * (d_r / vocab_size) if d_r < vocab_size else -10
                t_logp_data.append(t_lp)
                d_logp_data.append(d_lp)

                # Top-1 token ID (0 = target token, 1 = distractor, etc.)
                if t_r == 0:
                    top1_data.append(0)
                elif d_r == 0:
                    top1_data.append(1)
                else:
                    top1_data.append(random.randint(2, 10))

            entropy.append(pos_data)
            target_rank.append(t_rank_data)
            distractor_rank.append(d_rank_data)
            target_logp.append(t_logp_data)
            distractor_logp.append(d_logp_data)
            top1_id.append(top1_data)

    elif condition == "false_lead":
        # Distractor leads early, target catches up late; oscillation signature
        for layer_idx in range(n_layers):
            pos_data = []
            t_rank_data = []
            d_rank_data = []
            t_logp_data = []
            d_logp_data = []
            top1_data = []

            for pos in range(n_positions):
                layer_progress = layer_idx / n_layers

                # Three-phase: distractor dips, entropy recovers, target dips
                # This creates the oscillation signature: top-1 identity changes after
                # early confidence (distractor), then changes again to target
                phase1_end = 0.50   # Distractor strong
                phase2_start = 0.60 # Transition/recovery
                phase3_start = 0.75 # Target takes over

                if layer_progress < phase1_end:
                    # Early: distractor is confident (low entropy dip)
                    pos_ent = baseline_ents[layer_idx] * (0.4 + 0.6 * layer_progress)
                elif layer_progress < phase2_start:
                    # Middle: entropy rises as model reconsiders
                    pos_ent = baseline_ents[layer_idx] * (0.7 - 0.3 * (layer_progress - phase1_end) / (phase2_start - phase1_end))
                else:
                    # Late: target takes over, entropy drops again
                    pos_ent = baseline_ents[layer_idx] * (0.3 + 0.4 * (1 - layer_progress))

                pos_data.append(pos_ent)

                # Ranks follow three phases
                if layer_progress < phase1_end:
                    # Distractor leads strongly
                    d_r = max(0, int(vocab_size * 0.02))
                    t_r = max(2, int(vocab_size * 0.2 * (1 - layer_progress)))
                elif layer_progress < phase2_start:
                    # Transition: ranks get worse (uncertainty)
                    d_r = max(1, int(vocab_size * 0.15 * (1 - layer_progress)))
                    t_r = max(1, int(vocab_size * 0.18 * (1 - layer_progress)))
                else:
                    # Target wins decisively
                    t_r = 0
                    d_r = max(2, int(vocab_size * 0.25 * (1 - layer_progress)))

                t_rank_data.append(t_r)
                d_rank_data.append(d_r)

                # Log-probs follow ranks
                t_lp = -0.5 - 5 * (t_r / vocab_size) if t_r < vocab_size else -10
                d_lp = -2.0 - 5 * (d_r / vocab_size) if d_r < vocab_size else -10
                t_logp_data.append(t_lp)
                d_logp_data.append(d_lp)

                # Top-1 identity oscillation: distractor -> noise -> target
                if layer_progress < phase1_end:
                    # Phase 1: distractor wins
                    top1_data.append(1)
                elif layer_progress < phase2_start:
                    # Phase 2 (transition): oscillate between distractor and other tokens
                    # This creates the oscillation depth count
                    if random.random() < 0.6:
                        top1_data.append(random.randint(2, 8))  # noise
                    else:
                        top1_data.append(1)  # lingering distractor influence
                else:
                    # Phase 3: target is top-1
                    top1_data.append(0)

            entropy.append(pos_data)
            target_rank.append(t_rank_data)
            distractor_rank.append(d_rank_data)
            target_logp.append(t_logp_data)
            distractor_logp.append(d_logp_data)
            top1_id.append(top1_data)

    else:  # hard_control
        # Difficult (high entropy) but no tempting false target
        for layer_idx in range(n_layers):
            pos_data = []
            t_rank_data = []
            d_rank_data = []
            t_logp_data = []
            d_logp_data = []
            top1_data = []

            for pos in range(n_positions):
                layer_progress = layer_idx / n_layers
                # High entropy throughout, slow collapse
                pos_ent = baseline_ents[layer_idx] * (0.9 - 0.4 * layer_progress)
                pos_data.append(pos_ent)

                # Both ranks stay high (no easy distractor)
                if layer_progress > 0.75:
                    t_r = 0
                    d_r = max(100, int(vocab_size * 0.2))
                else:
                    t_r = max(50, int(vocab_size * 0.3 * (1 - layer_progress)))
                    d_r = max(100, int(vocab_size * 0.4 * (1 - layer_progress)))

                t_rank_data.append(t_r)
                d_rank_data.append(d_r)

                t_lp = -1.0 - 5 * (t_r / vocab_size) if t_r < vocab_size else -10
                d_lp = -5.0 - 5 * (d_r / vocab_size) if d_r < vocab_size else -10
                t_logp_data.append(t_lp)
                d_logp_data.append(d_lp)

                top1_data.append(random.randint(2, 20) if t_r > 0 else 0)

            entropy.append(pos_data)
            target_rank.append(t_rank_data)
            distractor_rank.append(d_rank_data)
            target_logp.append(t_logp_data)
            distractor_logp.append(d_logp_data)
            top1_id.append(top1_data)

    # Baseline entropy is position-independent (median of random samples)
    baseline_entropy = [be * random.uniform(0.95, 1.05) for be in baseline_ents]

    return {
        "layers": layers,
        "n_positions": n_positions,
        "entropy": entropy,
        "baseline_entropy": baseline_entropy,
        "target_rank": target_rank,
        "distractor_rank": distractor_rank,
        "target_logp": target_logp,
        "distractor_logp": distractor_logp,
        "top1_id": top1_id,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate mock traces for testing analysis pipeline."
    )
    ap.add_argument(
        "--out", default="out/traces_mock.json", help="Output JSON file"
    )
    ap.add_argument(
        "--n-per-family",
        type=int,
        default=3,
        help="Number of stimulus pairs per family (will create 2x for cond)",
    )
    ap.add_argument("--n-layers", type=int, default=25, help="Number of layers")
    ap.add_argument("--n-positions", type=int, default=20, help="Sequence length")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)

    # Stimuli families and example pairs
    families = {
        "factual": [
            ("fact_a", " Paris", " Amsterdam"),
            ("fact_b", " Tokyo", " Beijing"),
            ("fact_c", " Rome", " Athens"),
        ],
        "arithmetic": [
            ("arith_a", " 42", " 38"),
            ("arith_b", " 32", " 23"),
            ("arith_c", " 50", " 45"),
        ],
        "garden_path": [
            ("gp_a", " horse", " barn"),
            ("gp_b", " cat", " dog"),
            ("gp_c", " book", " shelf"),
        ],
    }

    records = []
    seed_offset = 0

    for family_name, pairs in families.items():
        for pair_id, target, distractor in pairs[:args.n_per_family]:
            for condition in ["straightforward", "false_lead"]:
                seed = args.seed + seed_offset
                seed_offset += 1

                trace = generate_layer_sequence(
                    args.n_layers,
                    args.n_positions,
                    condition,
                    seed,
                )

                record = {
                    "pair_id": pair_id,
                    "condition": condition,
                    "family": family_name,
                    "prompt": f"Test prompt for {pair_id} ({condition})",
                    "target": target,
                    "distractor": distractor,
                    "query_position": -1,
                    **trace,
                    "behavioral": {
                        "top5_tokens": [target, distractor, " other", " text", " here"],
                        "target_is_top1": True,  # Mock: assume straightforward is correct
                        "distractor_in_top5": True,
                    },
                }
                records.append(record)

            # Hard control: one per pair
            seed = args.seed + seed_offset
            seed_offset += 1
            trace = generate_layer_sequence(
                args.n_layers, args.n_positions, "hard_control", seed
            )
            record = {
                "pair_id": pair_id,
                "condition": "hard_control",
                "family": family_name,
                "prompt": f"Test prompt for {pair_id} (hard control)",
                "target": target,
                "distractor": distractor,
                "query_position": -1,
                **trace,
                "behavioral": {
                    "top5_tokens": [target, " other", " text", " here", " more"],
                    "target_is_top1": True,
                    "distractor_in_top5": False,
                },
            }
            records.append(record)

    # Write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"Generated {len(records)} mock traces -> {args.out}")
    print(f"  - {args.n_per_family} pairs × 3 families × 3 conditions = {args.n_per_family * 3 * 3} traces")


if __name__ == "__main__":
    main()
