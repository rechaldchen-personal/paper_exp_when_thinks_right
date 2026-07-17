#!/usr/bin/env python
"""
02_run_experiment_cot.py — Chain-of-thought variant for H4 testing.

H4: CoT externalizes revision, reducing internal oscillation on the same problems.

This script wraps the false-lead arithmetic problems in a CoT format and runs
the same trace collection as 02_run_experiment.py. Predictions: lower internal
oscillation, later commitment (revision externalizes to tokens).

Usage:
    python 02_run_experiment_cot.py --model Qwen/Qwen3-1.7B --lens out/lens.pt \
        --stimuli stimuli.json --out out/traces_cot.json

Generated traces have the same structure as 02_run_experiment.py, with an extra
"cot_enabled" flag to distinguish them in analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def cot_wrap_arithmetic(prompt: str) -> str:
    """Wrap an arithmetic prompt in CoT format.

    Examples:
        "calc: 4 + 17 * 2 =" ->
        "Let me work through this step by step.
         calc: 4 + 17 * 2 =
         First, I'll handle multiplication: 17 * 2 = 34.
         Then add: 4 + 34 ="
    """
    if "calc:" not in prompt:
        return prompt

    # For now, a simple CoT wrapper. In practice, you'd want task-specific
    # reasoning templates for each family.
    cot_prefix = (
        "Let me solve this step by step. "
        "I'll work through the order of operations carefully.\n"
    )
    # Replace the equals sign with a step-by-step prompt
    if prompt.endswith(" ="):
        prompt_trimmed = prompt[:-2]  # Remove " ="
        return f"{cot_prefix}{prompt_trimmed} =\nLet me compute this:"

    return prompt


def main():
    ap = argparse.ArgumentParser(
        description="CoT variant for H4 (commitment oscillation comparison)."
    )
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--lens", default="out/lens.pt")
    ap.add_argument("--stimuli", default="stimuli.json")
    ap.add_argument("--out", default="out/traces_cot.json")
    ap.add_argument(
        "--families",
        nargs="+",
        default=["arithmetic"],
        help="which stimulus families to wrap in CoT (default: arithmetic)",
    )
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["false_lead"],
        help="which conditions to wrap (default: false_lead)",
    )
    args = ap.parse_args()

    stimuli_data = json.loads(Path(args.stimuli).read_text())
    stimuli = stimuli_data["stimuli"]

    # Filter stimuli matching --families and --conditions
    filtered = []
    for s in stimuli:
        if s["family"] in args.families and s["condition"] in args.conditions:
            filtered.append(s)

    print(
        f"Wrapping {len(filtered)} stimuli in CoT "
        f"(families: {args.families}, conditions: {args.conditions})"
    )

    # Apply CoT wrapper
    for s in filtered:
        s["prompt_original"] = s["prompt"]
        s["prompt"] = cot_wrap_arithmetic(s["prompt"])
        s["cot_enabled"] = True

    # Add non-wrapped stimuli as-is
    non_wrapped = [s for s in stimuli if s not in filtered]
    print(f"Including {len(non_wrapped)} non-wrapped stimuli for comparison")

    # Combine
    all_stimuli = filtered + non_wrapped

    # The actual experiment would proceed as in 02_run_experiment.py:
    #   1. Load model, lens
    #   2. For each stimulus, run traces
    #   3. Output JSON
    #
    # For now, we just emit the wrapped stimuli as a test scaffold.
    # When you have GPU access, copy the trace-collection loop from
    # 02_run_experiment.py here and use `all_stimuli` instead of
    # the raw stimuli.

    print(f"\nTest output (wrapped stimuli, not yet traced):")
    print(json.dumps(all_stimuli[:2], indent=2))
    print(f"\n[Full experiment requires GPU; run after adapting the trace loop]")


if __name__ == "__main__":
    main()
