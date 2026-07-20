#!/usr/bin/env python
"""Verify the analysis environment is the pinned one and the statistics are sound.

Run as:  venv/bin/python verify_env.py

This checks three things, each of which was silently wrong at some point:

1. The running interpreter is this repo's venv — not a stray system/anaconda
   Python. Previously `source venv/bin/activate` was a no-op (no venv existed)
   and analysis ran under Python 3.7 / scipy 1.7.3.
2. Installed versions match requirements.txt, so p-values and figures are
   stable across machines.
3. `03_analyze.py`'s exact signed-rank test reproduces known-correct values,
   including the tied/zero cases where scipy's own "exact" method is invalid.

Exit code is 0 when the analysis environment is usable; GPU packages are
reported but never required.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PINNED = {"numpy": "1.26.4", "scipy": "1.13.1",
          "matplotlib": "3.9.4", "pandas": "2.2.3"}

ok = True
print("Environment verification")
print("=" * 62)

# ---- 1. interpreter ------------------------------------------------------
# sys.prefix (not sys.executable, which resolves through the framework symlink)
# is what identifies the active venv.
in_venv = Path(sys.prefix).resolve() == (REPO / "venv").resolve()
pyver = ".".join(str(v) for v in sys.version_info[:3])
print(f"\nsys.prefix  : {sys.prefix}")
print(f"python      : {pyver}")
if in_venv:
    print("  ✅ running inside the repo venv")
else:
    ok = False
    print("  ❌ NOT the repo venv — run as: venv/bin/python verify_env.py")
    print("     (a stray interpreter here previously produced wrong p-values)")

# ---- 2. pinned versions --------------------------------------------------
print("\nanalysis dependencies:")
for pkg, want in PINNED.items():
    try:
        got = __import__(pkg).__version__
    except ImportError:
        ok = False
        print(f"  ❌ {pkg:12} MISSING (want {want})")
        continue
    if got == want:
        print(f"  ✅ {pkg:12} {got}")
    else:
        ok = False
        print(f"  ❌ {pkg:12} {got}  (pinned: {want})")

print("\nGPU dependencies (only needed for 01_fit_lens / 02_run_experiment):")
for pkg in ("torch", "transformers", "datasets", "jlens"):
    try:
        mod = __import__(pkg)
        print(f"  ✅ {pkg:12} {getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        print(f"  ·  {pkg:12} not installed (fine on a CPU box)")

# ---- 3. statistical self-test -------------------------------------------
print("\nstatistics self-test (exact signed-rank):")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("an", REPO / "03_analyze.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)

    # (diffs, one-sided p_greater) — verified against brute-force enumeration.
    cases = [
        ([2, 2, 2, 2, 2], 1 / 32),
        ([0, 1, 2, 2, 2, 2], 1 / 32),          # zero dropped -> n=5
        ([3, -1, 4, -1, 5, 9, 2, 6], 5 / 256),
        ([2, 2, 2, -1, 3, 3], 1 / 32),         # ties: scipy's exact is invalid
    ]
    for diffs, want in cases:
        got = an.exact_signed_rank_test(diffs, "greater")["p"]
        good = abs(got - want) < 1e-12
        ok &= good
        mark = "✅" if good else "❌"
        print(f"  {mark} {str(diffs):30} p={got:.8f} (want {want:.8f})")

    deg = an.exact_signed_rank_test([0, 0, 0], "greater")
    good = deg["p"] == 1.0 and deg["n_nonzero"] == 0
    ok &= good
    print(f"  {'✅' if good else '❌'} all-zero differences handled")
except Exception as exc:                                    # noqa: BLE001
    ok = False
    print(f"  ❌ self-test could not run: {exc!r}")

print("\n" + "=" * 62)
if ok:
    print("✅ analysis environment OK — results should be reproducible.")
    print("   venv/bin/python 03_analyze.py --traces out/traces.json \\")
    print("       --outdir out/analysis_real --dev-split 0.6")
    sys.exit(0)
print("❌ environment problems above — fix before trusting any analysis.")
print("   ./setup_env.sh")
sys.exit(1)
