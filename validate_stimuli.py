#!/usr/bin/env python
"""validate_stimuli.py — enforce the stimulus design rules before any GPU run.

Every defect this checks for was present in the run-1 set and cost a full GPU
run's worth of unusable data. Run it after any edit to stimuli.json:

    venv/bin/python validate_stimuli.py --stimuli stimuli.json

Requires only the tokenizer (transformers, no torch), so it runs on CPU.

RULES
  R1 matched pair    within a pair_id, target and distractor are identical
                     across conditions, and both conditions are present.
                     Violated by all 28 arithmetic pairs in run 1, which
                     swapped target and distractor between conditions.
  R2 no leakage      the prompt must not end with the target or distractor
                     word. Violated by 15 garden-path items whose prompt ended
                     with the answer itself, leaving nothing to predict.
  R3 single token    target and distractor must each encode to exactly one
                     token. Qwen3 needs the leading-space form: ' fifteen' is
                     one token, 'fifteen' is three.
  R4 distinct        target != distractor.
  R5 query position  the resolved query index must be inside the prompt.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", default="stimuli.json")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--show", type=int, default=8,
                    help="max example failures to print per rule")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    stimuli = json.loads(Path(args.stimuli).read_text())["stimuli"]
    failures = defaultdict(list)

    # ---- per-item rules --------------------------------------------------
    for s in stimuli:
        tag = f"{s['pair_id']}/{s['condition']}"

        for field in ("target", "distractor"):
            ids = tok.encode(s[field], add_special_tokens=False)
            if len(ids) != 1:
                failures["R3 single token"].append(
                    f"{tag}: {field}={s[field]!r} -> {len(ids)} tokens "
                    f"({[tok.decode([i]) for i in ids]})")

        if s["target"] == s["distractor"]:
            failures["R4 distinct"].append(f"{tag}: both {s['target']!r}")

        prompt = s["prompt"].rstrip()
        for field in ("target", "distractor"):
            word = s[field].strip()
            if prompt.lower().endswith(word.lower()):
                failures["R2 no leakage"].append(
                    f"{tag}: prompt ends with {field} {word!r} — nothing to predict")

        # R6: in few-shot prompts the queried expression must not already be
        # demonstrated above, or the answer is sitting in the context. Caught a
        # real case where the query repeated a worked example verbatim.
        lines = [ln for ln in s["prompt"].split("\n") if ln.startswith("Q:")]
        if len(lines) > 1:
            query = lines[-1]
            expr = query.split("comes first:")[-1].strip()
            for shot in lines[:-1]:
                if expr and expr in shot:
                    failures["R6 no answer in context"].append(
                        f"{tag}: query {expr!r} already demonstrated above")
                    break

        n_pos = len(tok.encode(s["prompt"], add_special_tokens=False))
        if n_pos == 0 or not (-n_pos <= s["query_position"] < n_pos):
            failures["R5 query position"].append(
                f"{tag}: query_position={s['query_position']} vs {n_pos} tokens")

    # ---- pair-level rule -------------------------------------------------
    by_pair = defaultdict(dict)
    for s in stimuli:
        if s["condition"] != "hard_control":
            by_pair[s["pair_id"]][s["condition"]] = s

    for pid, conds in sorted(by_pair.items()):
        if set(conds) != {"straightforward", "false_lead"}:
            failures["R1 matched pair"].append(
                f"{pid}: conditions present = {sorted(conds)}")
            continue
        sf, fl = conds["straightforward"], conds["false_lead"]
        if sf["target"] != fl["target"]:
            failures["R1 matched pair"].append(
                f"{pid}: target differs — SF {sf['target']!r} vs FL {fl['target']!r}")
        if sf["distractor"] != fl["distractor"]:
            failures["R1 matched pair"].append(
                f"{pid}: distractor differs — SF {sf['distractor']!r} "
                f"vs FL {fl['distractor']!r}")
        if sf["prompt"] == fl["prompt"]:
            failures["R1 matched pair"].append(f"{pid}: prompts identical")

    # ---- report ----------------------------------------------------------
    from collections import Counter
    fam = Counter(s["family"] for s in stimuli)
    cond = Counter(s["condition"] for s in stimuli)
    print(f"{args.stimuli}: {len(stimuli)} stimuli, {len(by_pair)} pairs")
    print(f"  by family   : {dict(fam)}")
    print(f"  by condition: {dict(cond)}")
    print()

    if not failures:
        print("✅ all rules pass — safe to run on GPU")
        return 0

    total = sum(len(v) for v in failures.values())
    for rule in sorted(failures):
        items = failures[rule]
        print(f"❌ {rule}: {len(items)} failure(s)")
        for line in items[:args.show]:
            print(f"     {line}")
        if len(items) > args.show:
            print(f"     ... and {len(items) - args.show} more")
        print()
    print(f"{total} failure(s) — DO NOT run on GPU until these are fixed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
