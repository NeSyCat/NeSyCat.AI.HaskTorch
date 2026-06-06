# Tensor Variable Elimination (TVE) — design note & deferred future work

**Status:** deferred. Do this in the **Python/JAX port**, not in the Haskell prototype.
**Date:** 2026-06. **Audience:** us, later, when we generalize the marginalization engine.

---

## TL;DR — the decision

The Haskell prototype's marginalization engine is **already optimal for everything we run**:

- The additive / separable class (MNIST addition, multi-digit, weighted sums, counts, Binary's `iff`) routes through **`logConvolve`** (`Library/A_Categorical/Monads/LogVecExpect.hs`) — exact log-space **variable elimination on a treewidth-1 chain**. Peak `[B, maxSum+1]`, fully differentiable, normalized.
- Genuinely non-separable predicates fall back to the full-joint **`marginalize`** (kept as the correctness oracle), which is `O(∏ kᵢ)` and explodes.

A **general Tensor Variable Elimination engine** is the right way to handle non-separable-but-low-treewidth predicates exactly. But building it **in Haskell is overkill right now**:

1. **Nothing exercises it.** Every current example is separable → already served by `logConvolve` (the optimal contraction). A general engine would be dead code that reproduces `logConvolve` bit-for-bit.
2. **No ecosystem.** HaskTorch (untyped `Torch.*`) has **no `opt_einsum` / `cotengra`** (contraction-order optimizers) and **no `torch_semiring_einsum`** (a stable, differentiable log-semiring einsum). We'd hand-port all of it.
3. **The hard part is structure recovery.** `collectLeaves` flattens the formula AST to `([Tensor], [Int]->a)` — an *opaque reconstructor* — so the predicate looks like one dense arity-`n` factor. Recovering a *sparse* factorization from that is the real work, and it's fiddly.
4. **The theory wall is unavoidable anyway.** Exact marginalization is `#P`-complete; any exact method is `O(exp(treewidth))`. TVE only wins for predicates whose factor graph is *low-treewidth*, which the current benchmarks don't have.

**So: when we re-implement in Python/JAX (where the libraries exist and we'll target non-separable predicates), do TVE properly.** This note records the design so we don't re-derive it.

---

## Background — what marginalization does here

We train a neural classifier (e.g. an MNIST digit CNN) from only an **aggregate label** (e.g. the sum of two digits), by maximizing the probability that a logical predicate holds:

```
loss = -log P( predicate(latents) == observation | inputs )
```

`P(...)` is a **weighted model count** — a sum over all latent assignments satisfying the predicate, of the product of the per-latent neural probabilities. That sum is the marginalization. We do it **in log space** (logsumexp), **batched**, **differentiable** (autograd carries the gradient back to the CNN). This is the DeepProbLog-style *exact, probabilistic* reading — as opposed to LTN's *fuzzy, unnormalized* product-t-norm, which is cheaper but not a calibrated marginal.

### What's already built in Haskell (keep — it maps 1:1 to the Python engine)

| Haskell artifact | role | TVE interpretation |
|---|---|---|
| `logScatter` (`LogVecExpect.hs`) | per-bin logsumexp scatter | **one log-semiring contraction step** |
| `logConvolve` (`LogVecExpect.hs`) | fold additive leaves → `[B,maxSum+1]` | **TVE on a treewidth-1 chain** (the optimal order for `+`) |
| `marginalize` (`LogVecExpect.hs`) | full `[B, k₀…kₙ]` joint + logsumexp | **TVE at the worst order** (one dense factor over all vars) |
| `logNumDenConv` probe (`TensorBool.hs`) | detect `obs == base + Σ cᵢ(xᵢ)` | recovers the additive (chain) factor structure |
| `logNumDen` dispatch (`TensorBool.hs`) | `Just`→convolve / `Nothing`→marginalize | where a TVE branch would slot in |

So the Haskell prototype already contains **the two endpoints of TVE** (the treewidth-1 best case and the worst case) and **one contraction primitive**. The general engine is "the same algorithm with an arbitrary good elimination order."

---

## What TVE is, and why it's the right general engine

**Tensor Variable Elimination** (Obermeyer et al., *Tensor Variable Elimination for Plated Factor Graphs*, ICML 2019, [arXiv:1902.03210](https://arxiv.org/abs/1902.03210)) recasts exact sum-product inference so that **variable elimination = a scheduled sequence of tensor contractions (einsums) in a chosen semiring**:

- Each discrete latent is a tensor axis. A factor is a tensor over its scope (a CNN leaf is a `[B,k]` unary log-factor; the predicate is a constraint factor over the variables it couples; the observation is evidence).
- Eliminating a variable = multiply-in (add, in log) every factor touching it, then sum it out (logsumexp). One stable **log-semiring einsum** step.
- Swap the semiring's `(⊕, ⊗)` and the *same* contraction gives the marginal (sum-product / logsumexp), MAP (max-product), or samples.
- **Gradients are free**: reverse-mode autodiff through the forward contraction *is* backward message passing.
- **Plates = batch dimensions** (our `[B,…]` axis), product-reduced not summed — handled symbolically.

**Complexity:** `O(n · d^(w+1))` — exponential in the **treewidth** `w` of the chosen order, *linear* in the number of variables `n`. `logConvolve` is the `w = 1` case; `marginalize` is the `w = n−1` (worst) case. Finding the min-width order is NP-hard but cheap in practice for our tiny graphs (greedy min-fill, or `opt_einsum`'s exact DP).

**The key framing:** "evaluate the formula's factor graph in a good elimination order" *is* TVE, and it **subsumes `logConvolve`**. We're not adding a new paradigm — we're filling in the middle of a spectrum we already have the ends of.

---

## The plan for the Python/JAX port

When we re-implement, build the marginalization as TVE:

1. **Factor graph from the formula.** In Python we control the formula representation, so expose factor scopes directly (don't flatten to an opaque closure). Each latent → a `[B,k]` unary log-factor (the network's per-class logits); each predicate clause → a constraint factor over its variables; the observation → evidence.
2. **Contraction order.** Use **`opt_einsum`** (`contract_path` with `optimal`/`dp`/`greedy`) or **`cotengra`** (hypergraph partitioning + hyper-optimization, plus *slicing* to trade time for memory) to pick a near-min-treewidth order. **Compute the path once per formula shape** (the topology is fixed across batches/epochs) and reuse it.
3. **Execute in the log semiring.** Each step: broadcast-add the log-factors touching the eliminated variable, then `logsumexp` over its axis. Reference implementations:
   - **`torch_semiring_einsum`** (DuSell) — a differentiable, numerically-stable log-space einsum in PyTorch (block-chunked logsumexp-of-sums with a custom backward).
   - **Pyro `funsor`** — `naive_plated_einsum` / `TraceEnum_ELBO` (literally "Algorithm 1 of Obermeyer et al."), the closest published system to our exact-diff-batched goal.
   - **JAX** — `jax.numpy.einsum` won't do logsumexp directly; implement the log-semiring contraction with `logsumexp` + `jax.lax`/`vmap`, get autograd + `jit` for free. (`opt_einsum` is JAX-aware and can emit the path; you supply the per-step logsumexp kernel.)
4. **Slicing for memory** (cotengra's idea): if a peak intermediate is too big, fix one variable's index, loop over its `k` values, sum the results — exact, differentiable, time-for-memory.

This keeps the *exactness and normalization* that distinguish us from LTN, while replacing the `O(∏ kᵢ)` joint with a contraction whose cost is the formula's true treewidth.

---

## Concrete first target — the polynomial example (worked design)

A clean, honest demonstration where the full joint explodes but the structure is low-treewidth: **observe `z = d1·d2 + d3·d4`** (four MNIST digits 0..9, `z ∈ 0..162`). Non-additive (the products couple `d1–d2` and `d3–d4`), so the additive probe fails and the naive joint is `[B, 10⁴ · 163] ≈ 0.4 GB`.

**The "explicit factor" schedule (the insight to carry over):** pre-contract each product into a single message, which makes the *residual* additive:

```
p1 = contract d1,d2  via  (a,b) ↦ a*b      # message over {0,1,…,81}, treewidth-1 step
p2 = contract d3,d4  via  (a,b) ↦ a*b
result:  z == p1 + p2                       # ADDITIVE residual → ordinary convolution
```

- `contract` is one log-semiring step (= `logScatter` with the bin index = the host op `a*b`) — in Haskell this would be a `mergeWith :: (a→b→c) → m a → m b → m c` primitive (exactly the binary case of the parked `seqop` on branch `new-seqop`); in Python it's a single `logsumexp`-einsum over the `[B,10,10]` outer sum scattered by `a*b`.
- After the two product-contractions, `z == p1 + p2` is additive → reuse the convolution (`logConvolve`). **No change to the additive engine.**
- **The win:** peaks drop from ≈ 0.4 GB (full joint) to ≈ 1.5 MB (two `[B,100]`→`[B,37]` product contractions + a `[B,~163·37]` convolution).

This is also the general lesson: **TVE = contract the tightly-coupled sub-factors first (products), leaving a low-treewidth residual (the sum) for the cheap path.** The polynomial is the smallest example that shows it.

> Note: in the Haskell prototype this was fully designed and validated (the `mergeWith` pre-merge makes the residual pass the existing additive probe → routes to `logConvolve` with no dispatch change). It was **not implemented** — deferred here on purpose. The validation lives in the conversation/plan history; the primitive body is `git show new-seqop:Library/A_Categorical/DSL/Sem.hs` (the `SemMonad LogVec.seqop` binary case).

---

## TVE vs Knowledge Compilation (for later)

Both are exact, differentiable, and `O(exp(treewidth))`, and both hit the `#P` wall. They differ in **what structure they exploit below treewidth**:

- **TVE** = dense sum-product over the factor graph. Cost = *pure treewidth*. Cannot exploit determinism or context-specific independence (it sums over logically-impossible configs). But it stays entirely in tensors + autograd, batched, log-space — a direct generalization of `logConvolve`, **no external compiler**. → the right tool for **factorizable** predicates (products, comparisons, max/argmax, local/pairwise constraints).
- **Knowledge compilation** (d-DNNF/SDD; DeepProbLog/Semantic Loss; modern GPU eval via **KLay**, [arXiv:2410.11415](https://arxiv.org/abs/2410.11415)) compiles the constraint to a circuit that can be **exponentially smaller** than any treewidth-bounded form on **determinism-heavy** constraints (parity/XOR, exactly-one, global all-different). But it's a heavyweight `ground → CNF → compile → evaluate` pipeline that breaks out of autograd. → reach for it only when a predicate carries heavy determinism that blows up the dense joint while a circuit stays small.

**Beyond exact** (the `#P` frontier): A-NeSI (amortized neural WMC, [arXiv:2212.12393](https://arxiv.org/abs/2212.12393)), CTSketch (low-rank tensor sketching, [arXiv:2503.24123](https://arxiv.org/abs/2503.24123)), IndeCateR (score-function gradients, [arXiv:2311.12569](https://arxiv.org/abs/2311.12569)), and the CLT/saddlepoint closed form (our dormant `Giry` continuous reading). All trade exactness/calibration for tractability — only worth it where exact is hopeless.

---

## The theory wall (state it honestly)

Exact marginal inference = weighted model counting = **`#P`-complete**; the best any exact method achieves is **`O(n · exp(treewidth))`**, and even approximating the NeSy gradient *with guarantees* is NP-hard ([arXiv:2406.04472](https://arxiv.org/abs/2406.04472); Kwisthout [arXiv:1206.3240](https://arxiv.org/abs/1206.3240)). **TVE finds the best order for whatever treewidth the formula has — it cannot lower the treewidth.** A densely-coupled predicate (one monolithic `n`-ary factor) stays `O(dⁿ)`, identical to our current full joint. The payoff hinges entirely on the predicate decomposing into **many low-arity factors** — which is exactly why the factor-structure representation (point 1 of the port plan) is the whole game.

---

## References

- **TVE:** Obermeyer et al., ICML 2019 — [arXiv:1902.03210](https://arxiv.org/abs/1902.03210); Funsor — [arXiv:1910.10775](https://arxiv.org/abs/1910.10775).
- **Log-semiring einsum / contraction order:** `torch_semiring_einsum` (DuSell, bdusell.github.io/semiring-einsum); `opt_einsum` (Smith & Gray); `cotengra` / Gray & Kourtis — [arXiv:2002.01935](https://arxiv.org/abs/2002.01935); Generalized Distributive Law (Aji–McEliece 2000); Dechter bucket elimination.
- **Gradients = message passing:** Eisner 2016 ("Inside-outside and forward-backward … are just backprop"); Darwiche 2003.
- **Knowledge compilation & limits:** Darwiche & Marquis — [arXiv:1106.1819](https://arxiv.org/abs/1106.1819); DeepProbLog — [arXiv:1805.10872](https://arxiv.org/abs/1805.10872); KLay — [arXiv:2410.11415](https://arxiv.org/abs/2410.11415); Limits of Tractable Marginalization — [arXiv:2506.12020](https://arxiv.org/abs/2506.12020); tensor-network WMC — [arXiv:1908.04381](https://arxiv.org/abs/1908.04381).
- **Beyond exact:** A-NeSI [arXiv:2212.12393](https://arxiv.org/abs/2212.12393); CTSketch [arXiv:2503.24123](https://arxiv.org/abs/2503.24123); IndeCateR [arXiv:2311.12569](https://arxiv.org/abs/2311.12569).
