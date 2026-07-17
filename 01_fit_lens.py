#!/usr/bin/env python
"""
01_fit_lens.py — Fit a Jacobian lens for Qwen3 (or other open-weights model).

WHAT IT DOES:
    Computes per-layer Jacobian matrices J_ℓ (d_model × d_model) that map
    residual streams at layer ℓ into the final-layer logit basis.

    J_ℓ = average over prompts of (∂logits / ∂residual_ℓ)

    This enables efficient probing: apply J_ℓ to any hidden state to get
    predicted final-layer logits without running forward pass through upper layers.

REQUIRES: GPU (2-3h on single A100 for 1000 prompts × 128 tokens)

OUTPUT: lens.pt (~200MB), containing:
    - jacobians: dict[layer_id] → Tensor[d_model, d_model]
    - n_prompts: count of prompts used
    - d_model: hidden dimension
    - source_layers: which layers were fit

SETUP (first time):
    git clone https://github.com/anthropics/jacobian-lens
    pip install -e ./jacobian-lens
    pip install datasets

USAGE:
    # Full: 1000 prompts (2 hours on A100)
    python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 \
        --out out/lens_qwen3_1p7b.pt

    # Quick: 500 prompts (1 hour on A100, usually sufficient per Gurnee et al)
    python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 500 \
        --out out/lens_qwen3_1p7b_n500.pt

    # Resume if interrupted:
    python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 \
        --checkpoint out/fit_ckpt.pt --out out/lens_qwen3_1p7b.pt

API VERIFICATION: ✅ All signatures verified against jlens repo (2026-07-15)
    - jlens.from_hf(hf, tokenizer, ...) → HFLensModel
    - jlens.fit(model, prompts=..., checkpoint_path=...) → JacobianLens
"""

import argparse
from pathlib import Path


def load_fitting_corpus(tokenizer, n_prompts: int, seq_len: int = 128):
    """Pretraining-like web text, chunked to seq_len tokens, decoded back to text.

    The paper fits on 1000 x 128-token sequences from a generic web corpus.
    We use FineWeb (swap in C4/OpenWebText freely — choice is not critical).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    prompts, buf = [], []
    for ex in ds:
        buf.extend(tokenizer.encode(ex["text"]))
        while len(buf) >= seq_len:
            prompts.append(tokenizer.decode(buf[:seq_len]))
            buf = buf[seq_len:]
            if len(prompts) >= n_prompts:
                return prompts
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", default="out/lens.pt")
    ap.add_argument("--checkpoint", default="out/fit_ckpt.pt",
                    help="resumable fitting checkpoint")
    args = ap.parse_args()

    import torch
    import transformers
    import jlens  # official reference implementation

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)  # [API]

    print(f"Building fitting corpus ({args.n_prompts} x {args.seq_len} tokens) ...")
    prompts = load_fitting_corpus(tok, args.n_prompts, args.seq_len)
    print(f"  got {len(prompts)} sequences")

    print("Fitting lens (dominated by the model's backward pass; resumable) ...")
    lens = jlens.fit(model, prompts=prompts, checkpoint_path=args.checkpoint)  # [API]
    lens.save(args.out)
    print(f"Saved lens -> {args.out}")


if __name__ == "__main__":
    main()
