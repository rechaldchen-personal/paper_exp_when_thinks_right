#!/usr/bin/env python
"""build_stimuli.py — regenerate stimuli.json as strictly matched pairs.

WHY THIS EXISTS
    Run-1 diagnostics (2026-07-19) found three defects in the stimulus set that
    invalidated most of the paired analysis:

    1. Only 31/78 pairs held target AND distractor constant across conditions.
       Arithmetic (0/28) *swapped* them: straightforward asked (4+1)*2 -> ' ten'
       while false_lead asked 4+1*2 -> ' six'. Those are different questions, so
       a paired test confounds the framing effect with token identity/frequency.

    2. 15/23 garden-path items were degenerate: the prompt ended with the answer
       word itself ("...was the bread", target ' bread'), so there was no
       prediction to make at the query position.

    3. Arithmetic answers were unreachable in practice. Qwen3 tokenizes ' 10' as
       [220, 16, 15] — a bare space then digits — so the model's top-1 at the
       query position was token 220 (' ') on 55/56 items. That looked like
       failure but was the model correctly beginning a *digit* answer while the
       stimuli keyed on English number words.

DESIGN RULES ENFORCED HERE (all checked by validate_stimuli.py)
    R1  Within a pair, target and distractor are identical across conditions.
        Only the prompt framing changes. This is what makes the test paired.
    R2  The prompt must not end with the target or distractor word: the query
        position has to be a genuine prediction.
    R3  target and distractor must each be a single token with a leading space.
        Bare words are multi-token in Qwen3 ('fifteen' -> 3 tokens), so the
        leading-space form is mandatory.
    R4  Arithmetic answers are English number words in 1..20, all of which are
        single-token in the ' word' form, and the few-shot frame demonstrates
        the word format so the model does not open with a digit.

USAGE
    venv/bin/python build_stimuli.py --out stimuli.json
    venv/bin/python validate_stimuli.py --stimuli stimuli.json
"""

import argparse
import json
from pathlib import Path

NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty",
}

# Few-shot frame: establishes "Q: <expr>\nA: <word>" so the answer token is a
# leading-space number word rather than the model's default digit opening.
ARITH_SHOTS = (
    "Q: ( 1 + 2 ) * 2\nA: six\n"
    "Q: ( 3 + 1 ) * 4\nA: sixteen\n"
    "Q: ( 2 + 2 ) * 3\nA: twelve\n"
)
# The demonstrated expressions, excluded from the query set: reusing one would
# put the answer verbatim in the prompt.
SHOT_EXPRS = {(1, 2, 2), (3, 1, 4), (2, 2, 3)}


def arithmetic_pairs(limit=24):
    """Order-of-operations pairs where the ANSWER IS THE SAME in both conditions.

    target      = (a+b)*c   (the parenthesised reading, always the correct one)
    distractor  = a+b*c     (the precedence-confusion reading)

    straightforward: the expression is shown parenthesised from the start.
    false_lead:      the unparenthesised form is shown first, inviting the
                     distractor reading, then retracted in favour of the same
                     parenthesised expression. Target/distractor never move.

    Capped at `limit` pairs to keep the three families balanced: the paired
    tests pool across families, so an oversized arithmetic set would let the
    family with the least certain behavioural accuracy dominate the result.
    """
    out, seen = [], set()
    for a in range(1, 7):
        for b in range(1, 7):
            for c in range(2, 6):
                tgt, dis = (a + b) * c, a + b * c
                if tgt == dis or not (1 <= tgt <= 20 and 1 <= dis <= 20):
                    continue
                if (a, b, c) in SHOT_EXPRS:
                    continue
                key = (tgt, dis)
                if key in seen:          # keep answer-pairs distinct
                    continue
                seen.add(key)
                expr_p = f"( {a} + {b} ) * {c}"
                expr_u = f"{a} + {b} * {c}"
                pid = f"arith_{a}_{b}_{c}"
                common = dict(family="arithmetic", query_position=-1,
                              target=" " + NUM_WORDS[tgt],
                              distractor=" " + NUM_WORDS[dis])
                out.append(dict(pair_id=pid, condition="straightforward",
                                prompt=f"{ARITH_SHOTS}Q: {expr_p}\nA:", **common))
                out.append(dict(pair_id=pid, condition="false_lead",
                                prompt=(f"{ARITH_SHOTS}Q: {expr_u} — no wait, "
                                        f"the sum in brackets comes first: "
                                        f"{expr_p}\nA:"), **common))
                if len(out) >= 2 * limit:
                    return out
    return out


# (noun, competitor, reduced participle phrase, main predicate, category phrase,
#  plural?) -> classic reduced-relative garden path plus an explicit probe, so
# the query position is a real prediction rather than the answer word itself.
GP_ITEMS = [
    ("horse", "barn", "raced past the barn", "fell", "thing that fell", False),
    ("cat", "dog", "chased by the dog into the yard", "hid", "animal that hid", False),
    ("dog", "car", "hit by the car near the park", "limped", "animal that limped", False),
    ("boy", "dog", "chased by the dog", "ran fast", "person who ran", False),
    ("tree", "logger", "cut down in the forest by the logger", "fell", "thing that fell", False),
    ("package", "house", "delivered to the house on Monday", "was missing",
     "thing that was missing", False),
    ("sheep", "pen", "herded into the pen", "escaped", "animals that escaped", True),
    ("woman", "lawyer", "advised by the lawyer", "complained", "person who complained", False),
    ("man", "doctor", "examined by the doctor", "recovered", "person who recovered", False),
    ("child", "teacher", "praised by the teacher", "smiled", "person who smiled", False),
    ("soldier", "general", "ordered by the general", "retreated", "person who retreated", False),
    ("runner", "coach", "trained by the coach", "collapsed", "person who collapsed", False),
    ("bird", "cat", "startled by the cat", "flew away", "animal that flew away", False),
    ("boat", "storm", "damaged by the storm", "sank", "thing that sank", False),
    ("bridge", "flood", "weakened by the flood", "collapsed", "thing that collapsed", False),
    ("student", "professor", "questioned by the professor", "hesitated",
     "person who hesitated", False),
    ("patient", "nurse", "treated by the nurse", "improved", "person who improved", False),
    ("book", "editor", "reviewed by the editor", "sold well", "thing that sold well", False),
    ("house", "architect", "designed by the architect", "burned down",
     "thing that burned down", False),
    ("ship", "captain", "steered by the captain", "ran aground", "thing that ran aground", False),
    ("letter", "clerk", "misplaced by the clerk", "arrived late", "thing that arrived late", False),
    ("painting", "museum", "loaned to the museum", "was returned", "thing that was returned", False),
    ("singer", "producer", "recorded by the producer", "quit", "person who quit", False),
    ("watch", "tailor", "repaired by the tailor yesterday", "stopped", "thing that stopped", False),
]


def garden_path_pairs():
    """Reduced-relative garden paths. Target is the head noun in BOTH conditions.

    straightforward: full relative clause ("that was raced past the barn") —
                     unambiguous, no garden path.
    false_lead:      reduced relative ("raced past the barn") — the participle
                     is momentarily readable as the main verb, so the competitor
                     noun is briefly the better answer.
    """
    out = []
    for noun, comp, phrase, pred, category, plural in GP_ITEMS:
        rel = "that were" if plural else "that was"
        probe = f"The {category} " + ("were" if plural else "was the")
        pid = f"gp_{noun}"
        common = dict(family="garden_path", query_position=-1,
                      target=" " + noun, distractor=" " + comp)
        out.append(dict(pair_id=pid, condition="straightforward",
                        prompt=f"The {noun} {rel} {phrase} {pred}. {probe}",
                        **common))
        out.append(dict(pair_id=pid, condition="false_lead",
                        prompt=f"The {noun} {phrase} {pred}. {probe}",
                        **common))
    return out


# Genuinely hard retrieval with NO primed competitor. These isolate difficulty
# from false-lead temptation: if H1-H3 effects are about being misled rather
# than about being hard, hard controls should not show them.
HARD_CONTROLS = [
    ("hard_ctrl_currency", "Fact: The currency used in Switzerland is the", " franc", " euro"),
    ("hard_ctrl_president", "Fact: The first president of the United States was", " George", " Thomas"),
    ("hard_ctrl_painter", "Fact: The artist who painted the Mona Lisa was", " Leonardo", " Picasso"),
    ("hard_ctrl_element", "Fact: The lightest chemical element is", " hydrogen", " helium"),
    ("hard_ctrl_planet", "Fact: The largest planet in the solar system is", " Jupiter", " Saturn"),
    ("hard_ctrl_ocean", "Fact: The largest ocean on Earth is the", " Pacific", " Atlantic"),
    ("hard_ctrl_mountain", "Fact: The highest mountain on Earth is Mount", " Everest", " Fuji"),
    ("hard_ctrl_language", "Fact: The official language of Brazil is", " Portuguese", " Spanish"),
    ("hard_ctrl_metal", "Fact: The metal that is liquid at room temperature is", " mercury", " lead"),
    ("hard_ctrl_gas", "Fact: The most abundant gas in Earth's atmosphere is", " nitrogen", " oxygen"),
    ("hard_ctrl_river", "Fact: The longest river in Africa is the", " Nile", " Congo"),
    ("hard_ctrl_organ", "Fact: The organ that pumps blood around the body is the", " heart", " brain"),
]


def hard_controls():
    return [dict(pair_id=pid, condition="hard_control", family="factual",
                 prompt=prompt, target=tgt, distractor=dis, query_position=-1)
            for pid, prompt, tgt, dis in HARD_CONTROLS]


def carried_factual(src):
    """Carry over factual pairs from the previous set — the one family that was
    already correctly designed (24/24 matched). Only strictly matched pairs are
    kept, so nothing broken is inherited."""
    by_pair = {}
    for item in src:
        if item["family"] != "factual" or item["condition"] == "hard_control":
            continue
        by_pair.setdefault(item["pair_id"], {})[item["condition"]] = item
    out = []
    for pid, conds in sorted(by_pair.items()):
        if set(conds) != {"straightforward", "false_lead"}:
            continue
        sf, fl = conds["straightforward"], conds["false_lead"]
        if sf["target"] != fl["target"] or sf["distractor"] != fl["distractor"]:
            continue
        out += [sf, fl]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="stimuli.json",
                    help="previous stimuli file to carry factual items from")
    ap.add_argument("--out", default="stimuli.json")
    args = ap.parse_args()

    src = json.loads(Path(args.source).read_text())["stimuli"]

    stimuli = (carried_factual(src) + arithmetic_pairs()
               + garden_path_pairs() + hard_controls())

    payload = {
        "_notes": (
            "Strictly matched pairs: within a pair_id, target and distractor are "
            "identical across conditions and only the prompt framing changes. "
            "Regenerated by build_stimuli.py on 2026-07-19 after run-1 "
            "diagnostics; see that script's docstring for the three defects this "
            "replaces. Validate with validate_stimuli.py before any GPU run."
        ),
        "stimuli": stimuli,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    from collections import Counter
    fam = Counter(s["family"] for s in stimuli)
    cond = Counter(s["condition"] for s in stimuli)
    pairs = len({s["pair_id"] for s in stimuli if s["condition"] != "hard_control"})
    print(f"wrote {len(stimuli)} stimuli -> {args.out}")
    print(f"  pairs (2 conditions each): {pairs}")
    print(f"  by family   : {dict(fam)}")
    print(f"  by condition: {dict(cond)}")


if __name__ == "__main__":
    main()
