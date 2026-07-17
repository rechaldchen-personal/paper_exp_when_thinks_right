# API Resolution Guide for jlens

**Objective**: Map actual jlens API signatures before GPU experiments  
**Blocking**: Everything GPU-related  
**Estimated effort**: 2-3 hours (research + code updates)

---

## Pre-Work: Clone and Explore

```bash
# 1. Clone the official repo
git clone https://github.com/anthropics/jacobian-lens
cd jacobian-lens

# 2. Check structure
ls -la jlens/
# Expected: __init__.py, lens.py, model.py, etc.

# 3. Check what's exported from __init__.py
grep "from\|import" jlens/__init__.py

# 4. Check requirements
cat requirements.txt  # or pyproject.toml
# Need: transformers>=5.5, torch, ...
```

---

## Four API Call Sites to Resolve

### Call Site 1: Loading HF Model → jlens Model

**Current code** (01_fit_lens.py:71, 02_run_experiment.py:158):
```python
model = jlens.from_hf(hf, tok)  # [API]
```

**What to find**:
1. Open `jlens/model.py` or `jlens/__init__.py`
2. Search for `def from_hf`
3. Verify actual signature (parameters, return type)
4. Check: Does it need `model.eval()`? Does it detach grads?

**Expected questions**:
- Does `from_hf` exist or is it `Model.from_hf(...)`?
- What does it return? (e.g., `Model` class or a wrapped model?)
- Does it need tokenizer explicitly or infer from HF model?

**Resolution task**: Update line 71 (01_fit_lens.py) and line 158 (02_run_experiment.py)

---

### Call Site 2: Fitting the Lens

**Current code** (01_fit_lens.py:78):
```python
lens = jlens.fit(model, prompts=prompts, checkpoint_path=args.checkpoint)  # [API]
```

**What to find**:
1. Search for `def fit` in jlens/
2. Verify parameters (model, prompts, checkpoint_path, etc.)
3. Check return type (does it return a `JacobianLens` object or something else?)
4. Check: What is the actual loss? How does resuming work?

**Expected questions**:
- Exact parameter names: `prompts`, `prompt`, `data`?
- Is `checkpoint_path` the right name or is it `checkpoint`, `resume_from`?
- What does it return? (callable? dict? class instance?)

**Resolution task**: Update line 78 (01_fit_lens.py)

---

### Call Site 3: Loading a Fitted Lens

**Current code** (02_run_experiment.py:159):
```python
lens = jlens.JacobianLens.load(args.lens)  # [API]
```

**What to find**:
1. Search for `class JacobianLens` in jlens/
2. Find the load method: `def load(cls, path)`?
3. Check: Is it `.load()` or `.from_pretrained()`?
4. Does it auto-detect checkpoint format?

**Expected questions**:
- Is it `jlens.JacobianLens.load()` or `jlens.load()`?
- Is there a `.from_pretrained()` method for loading from HF?
- What does the returned object expose? (attributes/methods you'll use next)

**Resolution task**: Update line 159 (02_run_experiment.py)

---

### Call Site 4: Applying Lens & Getting Logits (Critical)

**Current code** (02_run_experiment.py:181–182):
```python
lens_logits, model_logits, _ = lens.apply(model, s["prompt"], positions=[...])  # [API]
```

**What to find**:
1. Search for `def apply` in the lens class
2. Check return signature (what are the three return values?)
3. Verify: Does `lens_logits` return a dict with layer indices?
4. Check: Is the shape `{layer: tensor[n_positions, vocab]}`?

**Expected questions**:
- Does `apply()` return (lens_logits, model_logits, something_else) or different?
- If it returns lens_logits, is it a dict keyed by layer index?
- What's the tensor shape? (Should be [n_positions, vocab])
- Does `positions` parameter accept a list or does it need to be indices?

**Resolution task**: Update lines 181–182 (02_run_experiment.py)

---

### 🔴 CRITICAL Call Site 5: unembed_fn (Random-Direction Baseline)

**Current code** (02_run_experiment.py:169–172):
```python
def unembed_fn(layer, h):
    h_final = lens.transport(layer, h.unsqueeze(0))  # [API] — CRITICAL
    h_final = hf.model.norm(h_final)
    return hf.lm_head(h_final)[0].float()
```

**Why this is critical**: The random-direction baseline is **essential** for all metrics (excess entropy, dissociation gap, etc.). Without this, you can't run the experiment.

**What to find**:
1. Search for how to transport an arbitrary vector through J_ℓ
2. Check if `lens.transport(layer, h)` exists
3. **If it doesn't exist**, check for:
   - Direct matrix access: `lens.J[layer]`? (then do `J @ h` manually)
   - Other transport methods: `lens.forward()`, `lens.project()`, etc.
   - Whether you need to compute J × h differently

**The three expected options**:

**Option A: Direct transport method**
```python
def unembed_fn(layer, h):
    h_final = lens.transport(layer, h.unsqueeze(0))
    h_final = hf.model.norm(h_final)
    return hf.lm_head(h_final)[0].float()
```

**Option B: Direct matrix access**
```python
def unembed_fn(layer, h):
    J = lens.J[layer]  # or lens.jacobian[layer]
    h_final = J @ h.unsqueeze(-1)  # or similar matrix multiply
    h_final = hf.model.norm(h_final)
    return hf.lm_head(h_final)[0].float()
```

**Option C: Different method name**
```python
def unembed_fn(layer, h):
    h_final = lens.project(layer, h)  # or other method name
    # ... rest same
```

**Investigation checklist**:
- [ ] Look for `transport` in lens class
- [ ] Look for `J` attribute (Jacobian matrix access)
- [ ] Look for `jacobian` attribute
- [ ] Search for other transport-like methods
- [ ] Check if there's example code in the repo using random directions

**Resolution task**: Update lines 169–172 (02_run_experiment.py)

---

## Secondary Checks

### Requirements & Versions
After identifying APIs, check:

```bash
# Check transformers version requirement
grep -i "transformers" requirements.txt pyproject.toml setup.py

# Expected: transformers>=5.5
# Check your GPU environment has this

# Check CUDA/torch compatibility
grep -i "torch" requirements.txt pyproject.toml

# Check if uv is used
cat pyproject.toml | grep "\[tool.uv\]"
```

**Why this matters**: Version mismatches are the #1 "works locally, breaks on cluster" failure

### Optional: Sparse Gradient-Pursuit

```bash
# Search for sparse pursuit code
find jlens/ -type f -name "*.py" -exec grep -l "sparse\|pursuit" {} \;
```

**For paper only**: If it exists, mention in methods/appendix for reproducibility. If not, that's fine—not needed for core experiment.

---

## Record Template

As you find each API, fill this in:

### Call Site 1: Loading HF Model
- **Actual function**: `jlens.from_hf(...)`  or similar
- **Signature**: `def from_hf(model, tokenizer, ...)`
- **Returns**: Type/class name
- **Code fix**: Update 01_fit_lens.py:71, 02_run_experiment.py:158
- **Verified**: [ ]

### Call Site 2: Fitting
- **Actual function**: `jlens.fit(...)` or similar
- **Signature**: Paste actual parameters
- **Returns**: Type/class name
- **Code fix**: Update 01_fit_lens.py:78
- **Verified**: [ ]

### Call Site 3: Loading Lens
- **Actual method**: `JacobianLens.load()` or `.from_pretrained()`
- **Signature**: Paste actual signature
- **Returns**: Type/class name
- **Code fix**: Update 02_run_experiment.py:159
- **Verified**: [ ]

### Call Site 4: Apply Lens
- **Actual method**: `lens.apply(model, prompt, positions=...)`
- **Return structure**: Dict keys? Tensor shapes?
- **Code fix**: Update 02_run_experiment.py:181–182
- **Verified**: [ ]

### Call Site 5: CRITICAL — unembed_fn
- **Transport method available**: Yes [ ] No [ ]
- **Actual implementation**:
  - Option A (transport): `lens.transport(layer, h)` [ ]
  - Option B (matrix): `lens.J[layer]`, `lens.jacobian[layer]` [ ]
  - Option C (other): `lens.project()`, etc. [ ]
- **Code fix**: Update 02_run_experiment.py:169–172
- **Verified**: [ ]

---

## After Resolution

1. **Update all four call sites** in the three scripts (01_fit_lens.py, 02_run_experiment.py)
2. **Test on CPU** (01_fit_lens.py should fail gracefully without GPU, but APIs should work)
3. **Document in code comments** exactly which version of jlens these APIs match
4. **Add to paper Methods** which version was used

---

## Risk Mitigation

**If API calls differ significantly from expected:**
- This is the #1 blocker for GPU runs
- Don't proceed with GPU experiments until APIs are verified
- The research shouldn't change; just the implementation wording

**If unembed_fn can't be resolved:**
- This is a showstopper (no random-direction baseline = no metrics)
- Contact jlens authors or check their repo issues for guidance
- May need to pivot to logit-lens-only (simpler approach, less novel)

---

## Timeline

- **Research**: Today (~2 hours)
- **Code updates**: Same session (~1 hour)
- **Testing on CPU**: Before GPU (~30 min)
- **Total**: 3–4 hours of focused work

This is the **critical path blocker**. Everything else waits for this.
