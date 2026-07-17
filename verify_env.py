#!/usr/bin/env python
"""
Verify that the virtual environment has all required dependencies installed.
Run this to check if your environment is ready before GPU testing.
"""

import sys

def check_import(package_name, display_name=None):
    """Try to import a package and report status."""
    if display_name is None:
        display_name = package_name

    try:
        mod = __import__(package_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {display_name:20} {version}")
        return True
    except ImportError:
        print(f"❌ {display_name:20} NOT INSTALLED")
        return False

print("Environment Verification")
print("=" * 60)
print()

# Check Python version
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"Python version: {py_version}")
print()

# Check core dependencies
print("Core Dependencies (CPU):")
print("-" * 60)
core_deps = [
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'Scikit-learn'),
    ('pandas', 'Pandas'),
    ('tqdm', 'TQDM'),
    ('requests', 'Requests'),
]

core_ok = all(check_import(pkg, display) for pkg, display in core_deps)

print()
print("GPU Dependencies (optional, needed for Phase 2+):")
print("-" * 60)
gpu_deps = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('datasets', 'Datasets'),
]

gpu_ok = all(check_import(pkg, display) for pkg, display in gpu_deps)

print()
print("=" * 60)
print()

if core_ok:
    print("✅ Core dependencies ready!")
    print("   You can run CPU-only phases (1, 4, 5, 6)")
else:
    print("❌ Some core dependencies missing!")
    print("   Run: pip install -r requirements.txt")

if gpu_ok:
    print("✅ GPU dependencies ready!")
    print("   You can run GPU phases (2, 3)")
else:
    print("⏳ GPU dependencies not installed")
    print("   When you have GPU access, run:")
    print("   pip install torch transformers datasets")

print()
print("=" * 60)

# Final status
if core_ok and gpu_ok:
    print("🚀 Environment is FULLY READY!")
    print("   You can run all GPU testing phases.")
    sys.exit(0)
elif core_ok:
    print("⚠️  Environment is PARTIALLY READY!")
    print("   Core work is ready; waiting for GPU dependencies.")
    sys.exit(1)
else:
    print("❌ Environment needs setup!")
    print("   Run: source venv/bin/activate && pip install -r requirements.txt")
    sys.exit(2)
