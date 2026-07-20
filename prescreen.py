#!/usr/bin/env python
"""prescreen.py — behavioural gate that must pass before any hypothesis test.

Implements the stop condition in PRE_REGISTRATION_AMENDMENT.md §8. Run it on
freshly collected traces and read the verdict BEFORE running 03_analyze.py:

    venv/bin/python prescreen.py --traces out/traces_run2.json

If the model cannot answer a family's straightforward condition, the lens
metrics for that family are measuring nothing: l_star is undefined when the
target never reaches top-1, so those pairs silently drop out of the paired
tests and the surviving n is both tiny and non-random.

This is exactly what went wrong in run 1 — arithmetic scored 1/56 on the
straightforward condition and was analyzed anyway, contributing 0/13 defined
commitment layers while still consuming half the stimulus set.

Exit code 0 = safe to analyze, 1 = fix stimuli and recollect traces.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="out/traces_run2.json")
    ap.add_argument("--min-accuracy", type=float, default=0.5,
                    help="required straightforward top-1 accuracy per family")
    ap.add_argument("--show", type=int, default=5,
                    help="example failures to print per family")
    args = ap.parse_args()

    records = json.loads(Path(args.traces).read_text())

    acc = defaultdict(lambda: [0, 0])          # family -> [correct, total]
    tempted = defaultdict(lambda: [0, 0])      # family -> [distractor top5, total]
    examples = defaultdict(list)

    for r in records:
        fam, cond = r["family"], r["condition"]
        beh = r["behavioral"]
        if cond == "straightforward":
            acc[fam][1] += 1
            if beh["target_is_top1"]:
                acc[fam][0] += 1
            else:
                examples[fam].append(
                    f"{r['pair_id']}: want {r['target']!r}, got "
                    f"{beh['top5_tokens'][0]!r} (top5 {beh['top5_tokens']})")
        elif cond == "false_lead":
            tempted[fam][1] += 1
            if beh["distractor_in_top5"]:
                tempted[fam][0] += 1

    print(f"{args.traces}: {len(records)} traces\n")
    print("straightforward top-1 accuracy (must clear "
          f"{args.min_accuracy:.0%} per family):")
    ok = True
    for fam in sorted(acc):
        c, n = acc[fam]
        rate = c / n if n else 0.0
        good = rate >= args.min_accuracy
        ok &= good
        print(f"  {'✅' if good else '❌'} {fam:12} {c:3d}/{n:<3d}  {rate:6.1%}")

    print("\nfalse-lead temptation (distractor in final top-5 — diagnostic, "
          "not a gate):")
    for fam in sorted(tempted):
        c, n = tempted[fam]
        print(f"     {fam:12} {c:3d}/{n:<3d}  {c / n if n else 0:6.1%}")

    for fam in sorted(examples):
        c, n = acc[fam]
        if n and c / n < args.min_accuracy:
            print(f"\n  failing examples — {fam}:")
            for line in examples[fam][:args.show]:
                print(f"     {line}")

    print()
    if ok:
        print("✅ pre-screen PASSED — safe to run 03_analyze.py")
        return 0
    print("❌ pre-screen FAILED — per amendment §8, fix the stimuli for the")
    print("   failing family and recollect traces. Do NOT analyze around it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
