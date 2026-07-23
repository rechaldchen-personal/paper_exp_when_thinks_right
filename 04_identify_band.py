#!/usr/bin/env python
"""
04_identify_band.py — Per-model workspace band identification (Appendix C).

WHAT IT DOES:
    Implements the per-model band identification promised in Methods §4.1 and
    experiments/workspace_band_guide.md, on a held-out text corpus (NOT the
    corpus used to fit the lens — see --skip-prompts). For each layer,
    computes three signals (math in band_analysis.py, unit-tested in
    test_band_analysis.py):
      1. Excess kurtosis of the lens-readout entropy distribution
      2. Layer-to-layer rank autocorrelation (this script's operationalization
         of the guide's ambiguous pseudocode — see band_analysis.py docstring)
      3. Next-token top-1 accuracy
    then combines them into a composite score and suggests a band as the
    longest contiguous layer run above threshold. This is a STARTING POINT
    for the human judgment call in Appendix C — always look at the plot.

REQUIRES: GPU. Far cheaper than lens fitting: ~5-15 min for 200 held-out
    prompts (forward passes only, no backward pass, unlike lens fitting).

USAGE:
    # jlens readout (primary; needs the fitted lens)
    python 04_identify_band.py --model Qwen/Qwen3-1.7B --readout jlens \
        --lens out/lens_qwen3_1p7b.pt --n-prompts 200 \
        --out out/band_identification_jlens.json

    # logit-lens readout (cross-check; no --lens needed)
    python 04_identify_band.py --model Qwen/Qwen3-1.7B --readout logit_lens \
        --n-prompts 200 --out out/band_identification_logitlens.json

OUTPUT:
    <out>.json  — per-layer curves + suggested band (layers and fraction)
    out/figures/band_identification_<readout>.png — 3-panel plot

The readout dispatch (jlens vs logit_lens) mirrors 02_run_experiment.py's
per-stimulus loop intentionally rather than sharing a helper function: the
two readouts have genuinely different call conventions (jlens.apply() wants
prompt text, apply_logit_lens() wants token ids), so unifying them behind one
signature would obscure more than it simplifies. If you change the readout
dispatch logic in one script, check the other.
"""

import argparse
import json
from pathlib import Path


def load_heldout_corpus(tokenizer, n_prompts: int, seq_len: int = 128,
                        skip: int = 1000):
    """FineWeb chunks, skipping the first `skip` to avoid overlap with the
    lens-fitting corpus (01_fit_lens.py --n-prompts default is 1000; keep
    `skip` >= whatever --n-prompts was used to fit the lens being evaluated).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    prompts, buf, skipped = [], [], 0
    for ex in ds:
        buf.extend(tokenizer.encode(ex["text"]))
        while len(buf) >= seq_len:
            chunk, buf = buf[:seq_len], buf[seq_len:]
            if skipped < skip:
                skipped += 1
                continue
            prompts.append(tokenizer.decode(chunk))
            if len(prompts) >= n_prompts:
                return prompts
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--readout", choices=["jlens", "logit_lens"], default="jlens")
    ap.add_argument("--lens", default="out/lens.pt",
                    help="required for --readout jlens")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--skip-prompts", type=int, default=1000,
                    help="skip this many FineWeb chunks before sampling, to "
                         "avoid overlap with the lens-fitting corpus")
    ap.add_argument("--threshold-pct", type=float, default=40.0,
                    help="percentile of the composite score defining the band")
    ap.add_argument("--out", default="out/band_identification.json")
    args = ap.parse_args()

    import torch
    import transformers
    if args.readout == "jlens":
        import jlens
    else:
        from lens_utils import apply_logit_lens, logit_lens_unembed_fn  # noqa: F401 (unembed_fn unused here)

    from band_analysis import identify_band

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda()
    hf.eval()
    device = next(hf.parameters()).device

    if args.readout == "jlens":
        model = jlens.from_hf(hf, tok)                              # [API]
        lens = jlens.JacobianLens.load(args.lens)                   # [API]

    print(f"Loading held-out corpus ({args.n_prompts} x {args.seq_len} tokens, "
          f"skipping first {args.skip_prompts}) ...")
    prompts = load_heldout_corpus(tok, args.n_prompts, args.seq_len,
                                  args.skip_prompts)
    print(f"  got {len(prompts)} held-out sequences")

    entropy_by_layer, rank_by_layer, correct_by_layer = {}, {}, {}
    layers = None

    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.seq_len).input_ids.to(device)
        n_pos = ids.shape[1]
        if n_pos < 2:
            continue  # need at least one (position, next-token) pair
        positions = list(range(n_pos))

        # readout dispatch — mirrors 02_run_experiment.py; see module docstring
        if args.readout == "jlens":
            lens_logits, _, _ = lens.apply(model, prompt, positions=positions)  # [API]
        else:
            lens_logits, _, _ = apply_logit_lens(hf, ids, positions=positions)

        if layers is None:
            layers = sorted(lens_logits.keys())
            for L in layers:
                entropy_by_layer[L] = []
                rank_by_layer[L] = []
                correct_by_layer[L] = []

        next_token_ids = ids[0, 1:].tolist()  # true next token for positions 0..n_pos-2
        for L in layers:
            ll = lens_logits[L].float()  # (n_pos, vocab)
            logp = torch.log_softmax(ll, dim=-1)
            H = -(logp.exp() * logp).sum(-1)  # (n_pos,) entropy in nats
            argmax = ll.argmax(dim=-1)         # (n_pos,)

            # only positions 0..n_pos-2 have a defined next token
            for p in range(n_pos - 1):
                true_next = next_token_ids[p]
                entropy_by_layer[L].append(float(H[p]))
                rank_by_layer[L].append(
                    int((ll[p] > ll[p, true_next]).sum().item()))
                correct_by_layer[L].append(int(argmax[p].item() == true_next))

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(prompts)} prompts")

    print("Computing kurtosis / autocorrelation / accuracy curves ...")
    result = identify_band(layers, entropy_by_layer, rank_by_layer,
                           correct_by_layer, threshold_pct=args.threshold_pct)
    result["model"] = args.model
    result["readout"] = args.readout
    result["n_prompts"] = len(prompts)
    result["n_layers"] = len(layers)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nSuggested band (layers): {result['suggested_band_layers']}")
    print(f"Suggested band (fraction): {result['suggested_band_fraction']}")
    print(f"Wrote {args.out}")

    # ---- plot ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    curves = [("kurtosis", "Excess kurtosis (entropy)"),
             ("autocorrelation", "Layer-to-layer rank autocorrelation"),
             ("accuracy", "Next-token top-1 accuracy")]
    band = result["suggested_band_layers"]
    for ax, (key, title) in zip(axes, curves):
        ax.plot(layers, result[key], marker="o", markersize=3)
        if band:
            ax.axvspan(band[0], band[1], alpha=0.15, color="tab:orange")
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Layer index")
    fig.suptitle(f"Band identification — {args.model} ({args.readout})\n"
                f"suggested band: layers {band}, "
                f"fraction {result['suggested_band_fraction']}")
    fig.tight_layout()
    plot_path = Path("out/figures") / f"band_identification_{args.readout}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
