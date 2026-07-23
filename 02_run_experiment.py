#!/usr/bin/env python
"""
02_run_experiment.py — Collect lens traces on all stimuli.

WHAT IT DOES:
    For each stimulus:
    1. Tokenizes prompt and identifies query position
    2. Runs forward pass, records residuals at each layer
    3. Applies a readout at every layer (--readout jlens|logit_lens):
       - jlens: J_ℓ @ residual_ℓ → predicted logits (fitted Jacobian lens)
       - logit_lens: residual_ℓ → norm → lm_head directly (identity
         transport, no fitted lens needed) — the robustness readout from
         Methods §4.7 / Appendix D ("if H1-H3 hold under both J-lens and
         logit-lens, the findings are not a J-lens-specific artifact")
    4. Records entropy, target/distractor ranks, top-1 oscillations
    5. Computes a random-direction baseline under the SAME readout
    6. Validates behavior (does model answer correctly?)

OUTPUT: traces*.json with one record per stimulus, each containing:
    - lens_method: "jlens" or "logit_lens" (which readout produced this record)
    - layers: [0, 1, ..., n_layers-1] covered by the readout
    - n_positions: token count in prompt
    - entropy[L][T]: per-(layer, token) entropy from the readout's logits
    - baseline_entropy[L]: random-direction entropy (position-independent)
    - target_rank[L][T], distractor_rank[L][T]: 0-based ranks
    - target_logp[L][T], distractor_logp[L][T]: log probabilities
    - top1_id[L][T]: argmax token id (for oscillation detection)
    - behavioral: validation (target_is_top1, distractor_in_top5)
    Both readouts produce records with an identical schema, so
    03_analyze.py runs on either trace file unmodified — that's what makes
    the two runs directly comparable for the robustness check.

REQUIRES: GPU (jlens: ~20 min on A100; logit_lens: similar, and does not
need a fitted lens file or the jlens package at all)

USAGE:
    # Validate tokenization (CPU-only, no GPU)
    python 02_run_experiment.py --validate

    # Primary readout (fitted Jacobian lens, GPU required)
    python 02_run_experiment.py \
        --model Qwen/Qwen3-1.7B --readout jlens \
        --lens out/lens_qwen3_1p7b.pt \
        --out out/traces_run2.json

    # Robustness readout (identity transport, GPU required, no --lens needed)
    python 02_run_experiment.py \
        --model Qwen/Qwen3-1.7B --readout logit_lens \
        --out out/traces_run2_logitlens.json

    # Then compare: run 03_analyze.py on each trace file and check H1-H3
    # hold in the same direction under both (Appendix D).

STIMULI: see stimuli.json (validate_stimuli.py enforces the design rules)
    - Straightforward & false-lead conditions (tests hypotheses H1–H3)
    - Hard controls (validates specificity to false-lead, not difficulty)

API VERIFICATION: ✅ jlens signatures verified (2026-07-15)
    - jlens.from_hf(hf, tokenizer) → HFLensModel
    - JacobianLens.load(path) → JacobianLens
    - lens.apply(model, prompt, positions=...) → (lens_logits_dict, model_logits, input_ids)
    - lens.transport(residual, layer) → transported_logits (for random baseline)
If your installed jlens returns a different shape, adapt the jlens branch
below. --readout logit_lens has no jlens dependency at all (lens_utils.py).
The random-direction baseline needs raw residual activations; both readouts
obtain them via standard forward hooks (capture_residuals / lens_utils),
independent of jlens internals.
"""

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def entropy_from_logits(logits):
    """Shannon entropy (nats) of softmax(logits). logits: (..., vocab)."""
    import torch
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def rank_of(logits, token_id):
    """0-based rank of token_id in descending logits. logits: (vocab,)."""
    return int((logits > logits[token_id]).sum().item())


def single_token_id(tokenizer, s):
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return ids[0]


def capture_residuals(hf_model, input_ids):
    """Forward pass capturing the residual stream at every layer.

    Returns dict {layer_idx: tensor[seq, d_model]} plus final logits.
    Works for standard HF decoder architectures exposing .model.layers;
    adjust `blocks` accessor for exotic architectures.
    """
    import torch

    acts = {}
    hooks = []
    blocks = hf_model.model.layers  # LlamaModel/QwenModel-style

    def mk_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[idx] = h.detach()[0]  # (seq, d_model), batch=1
        return hook

    for i, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(mk_hook(i)))
    with torch.no_grad():
        out = hf_model(input_ids)
    for h in hooks:
        h.remove()
    return acts, out.logits[0]  # (seq, vocab)


def random_baseline_entropy(lens, layer, ref_norm, unembed_fn, n_samples, d_model,
                            device, dtype):
    """Median lens-readout entropy of random directions with norm == ref_norm.

    Following the paper's convention of reporting metrics in EXCESS of a
    random-direction control, so early-layer lens degeneracy and vocabulary
    priors don't masquerade as signal.

    unembed_fn(layer, h) must return lens logits for arbitrary vectors h.
    We construct it in main() from the lens object; if your jlens version
    exposes lens matrices differently, adjust there (marked [API]).
    """
    import torch
    ents = []
    for _ in range(n_samples):
        v = torch.randn(d_model, device=device, dtype=torch.float32)
        v = v / v.norm() * ref_norm
        logits = unembed_fn(layer, v.to(dtype))
        ents.append(float(entropy_from_logits(logits)))
    ents.sort()
    return ents[len(ents) // 2]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--readout", choices=["jlens", "logit_lens"], default="jlens",
                    help="jlens: fitted Jacobian lens (primary). "
                         "logit_lens: identity transport, no --lens/jlens needed "
                         "(robustness readout, Methods §4.7 / Appendix D)")
    ap.add_argument("--lens", default="out/lens.pt",
                    help="required for --readout jlens; ignored for logit_lens")
    ap.add_argument("--stimuli", default="stimuli.json")
    ap.add_argument("--out", default="out/traces.json")
    ap.add_argument("--baseline-samples", type=int, default=32)
    ap.add_argument("--validate", action="store_true",
                    help="only check that targets/distractors are single tokens")
    args = ap.parse_args()

    import transformers  # tokenizer-only; safe without torch/GPU

    stimuli = json.loads(Path(args.stimuli).read_text())["stimuli"]
    tok = transformers.AutoTokenizer.from_pretrained(args.model)

    # ---- validation pass (always do this before burning GPU time) ----------
    # torch/jlens are NOT imported yet, so --validate genuinely runs CPU-only
    # as documented, rather than failing on a missing torch install. (Prefer
    # validate_stimuli.py for day-to-day use — it checks 7 design rules, not
    # just tokenization — this stays as a quick pre-GPU sanity check.)
    bad = []
    for s in stimuli:
        for field in ("target", "distractor"):
            if single_token_id(tok, s[field]) is None:
                bad.append((s["pair_id"], s["condition"], field, s[field]))
    if bad:
        print("MULTI-TOKEN targets/distractors found — fix stimuli.json first:")
        for b in bad:
            print("  ", b)
        if args.validate:
            return
        raise SystemExit(1)
    print(f"Tokenization OK: {len(stimuli)} stimuli, all targets single-token.")
    if args.validate:
        return

    import torch
    if args.readout == "jlens":
        import jlens
    else:
        from lens_utils import apply_logit_lens, logit_lens_unembed_fn

    # ---- model + lens -------------------------------------------------------
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda()
    hf.eval()
    d_model = hf.config.hidden_size
    device, dtype = next(hf.parameters()).device, next(hf.parameters()).dtype

    if args.readout == "jlens":
        model = jlens.from_hf(hf, tok)                                   # [API]
        lens = jlens.JacobianLens.load(args.lens)                        # [API]

        # [API] Build an "apply lens to an arbitrary vector" function.
        # Preferred: use lens matrices directly if exposed, e.g. lens.J[layer].
        # Fallback shown here assumes lens.transport(layer, h) -> final-basis
        # vector and model unembedding via hf.lm_head / final norm. Inspect
        # your jlens version (jlens/*.py) and pick whichever is available;
        # only this closure needs to change.
        def unembed_fn(layer, h):
            h_final = lens.transport(h.unsqueeze(0).float(), layer)      # [API]
            h_final = hf.model.norm(h_final.to(dtype))
            return hf.lm_head(h_final)[0].float()
    else:
        # Identity transport: no fitted lens, no jlens dependency. `lens` stays
        # None — random_baseline_entropy()'s `lens` parameter is unused inside
        # the function (unembed_fn already closes over whatever it needs) but
        # the call site below references the name unconditionally.
        lens = None
        unembed_fn = logit_lens_unembed_fn(hf)

    records = []
    for s in stimuli:
        ids = tok(s["prompt"], return_tensors="pt").input_ids.to(device)
        n_pos = ids.shape[1]
        positions = list(range(n_pos))

        if args.readout == "jlens":
            # lens readouts at every position, every layer the lens covers
            lens_logits, model_logits, _ = lens.apply(                   # [API]
                model, s["prompt"], positions=positions
            )
            layers = sorted(lens_logits.keys())
            # residual norms per layer (for norm-matched random baseline)
            acts, final_logits = capture_residuals(hf, ids)
        else:
            # identity-transport readout; also returns residuals (norm-matched
            # baseline), so no separate capture_residuals() call is needed
            lens_logits, final_logits, acts = apply_logit_lens(
                hf, ids, positions=positions
            )
            layers = sorted(lens_logits.keys())

        t_id = single_token_id(tok, s["target"])
        d_id = single_token_id(tok, s["distractor"])

        rec = {k: s[k] for k in
               ("pair_id", "condition", "family", "prompt", "target",
                "distractor", "query_position")}
        rec.update({
            "lens_method": args.readout,
            "layers": layers, "n_positions": n_pos,
            "entropy": [], "baseline_entropy": [],
            "target_rank": [], "distractor_rank": [],
            "target_logp": [], "distractor_logp": [],
            "top1_id": [],
        })

        for L in layers:
            ll = lens_logits[L].float()          # (n_pos, vocab)
            logp = torch.log_softmax(ll, dim=-1)
            H = entropy_from_logits(ll)          # (n_pos,)

            rec["entropy"].append([float(x) for x in H])
            rec["target_rank"].append([rank_of(ll[p], t_id) for p in range(n_pos)])
            rec["distractor_rank"].append([rank_of(ll[p], d_id) for p in range(n_pos)])
            rec["target_logp"].append([float(logp[p, t_id]) for p in range(n_pos)])
            rec["distractor_logp"].append([float(logp[p, d_id]) for p in range(n_pos)])
            rec["top1_id"].append([int(ll[p].argmax()) for p in range(n_pos)])

            ref_norm = float(acts[L].float().norm(dim=-1).median()) \
                if L in acts else float(acts[max(acts)].float().norm(dim=-1).median())
            rec["baseline_entropy"].append(
                random_baseline_entropy(lens, L, ref_norm, unembed_fn,
                                        args.baseline_samples, d_model,
                                        device, dtype))

        q = s["query_position"] % n_pos
        top5 = final_logits[q].topk(5).indices.tolist()
        rec["behavioral"] = {
            "top5_tokens": [tok.decode([i]) for i in top5],
            "target_is_top1": top5[0] == t_id,
            "distractor_in_top5": d_id in top5,
        }
        records.append(rec)
        print(f"done: {s['pair_id']:>18s} / {s['condition']:<15s} "
              f"top1={rec['behavioral']['top5_tokens'][0]!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records))
    print(f"\nWrote {len(records)} traces (readout={args.readout}) -> {args.out}")


if __name__ == "__main__":
    main()
