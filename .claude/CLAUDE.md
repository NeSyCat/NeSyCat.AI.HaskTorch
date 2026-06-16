---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Haskell implementation of the **NeSyCat** neurosymbolic framework using HaskTorch (untyped `Torch.*` bindings to libtorch). The code is a typed tensor-category realization of the theory in the companion paper repo (`../NeSyCat Papers/`).

## Build & Run Commands

Run an example by its **exact folder name** with the `./nesycat` wrapper — it regenerates the dispatcher from the `Examples/` folders (auto-discovery), runs hpack, builds, and runs:

```bash
./nesycat MnistAddition 1   # ./nesycat <ExampleName> [n]   (n = runs to average; n=1 prints the loss curve)
./nesycat Binary 10         # data ships in the example; Examples/MnistAddition/E_Data/get-mnist.sh refetches MNIST
./nesycat Binary 1 +RTS -s  # RTS stats (exe built with -rtsopts)

cabal build all             # plain build; the cabal is hpack-generated (~/.cabal/bin/hpack --force regenerates it)
```

Adding an `Examples/<Name>/` folder is **auto-registered** on the next `./nesycat` run — no manual step. The dispatcher `Library/Run.hs` is generated from the folder list (don't hand-edit it). `package.yaml` (hpack) generates `nesycat-hasktorch.cabal` (committed).

Requires `hasktorch` + libtorch. HLS auto-detects the cabal cradle (reads the generated `.cabal`). `hpack` lives at `~/.cabal/bin/hpack`.

## Architecture: by-example over a shared library

The repo is organized **by example**, not by layer. Two folders at the repo root hold everything — `Library/` (shared) and `Examples/` (per example):

- **`Library/`** — the shared, reusable framework, keeping the A–G layer names:
  - `A_Categorical.Monads.*` — the monads the readings run in: `Dist` (finitely-supported probability distributions) and `LogTens` (finitely-supported, NON-normalized, log-space measures — the batched GPU/autograd sibling of `Dist`, whose Kleisli bind is the log-space convolution), each with its `*Expect` evaluator; plus `Giry` (the continuous monad, defined but unused, ready as a 3rd reading). **There are NO universes, no `Framework`/`Universe` class, no `GeomU`/`MeasU` tags, no `eta`/`mu`** — the framework parametrizes over the monad `m` *directly* (a "reading" is just a choice of `m`, e.g. `@Dist` / `@LogTens`). The ONLY inter-monad structure is `A_Categorical.Monads.Bridge`: `encode :: [a] -> Tensor -> LogTens a` (the batched `Dist⇒LogTens` embedding — lifts observations/inputs given as a `[B,k]` one-hot/probability tensor into a leaf) and `decode :: LogTens a -> Dist a` (the `LogTens⇒Dist` readout — softmax a leaf, per example) — a section/retraction pair (`decode . encode = id`). (There is no `Category/` subfolder; the monads + bridge live directly under `A_Categorical/Monads/`.)
  - `B_Logical.*` — the truth algebra. `Signature.TwoMonBLat tau` = the **connectives** (∨,∧,⊕,⊗,¬,→ on the truth type `tau`) — *monad-free* (one Haskell category; the connectives read the same in every monad). `Signature.A2MonBLat m tau` = the **quantifiers/aggregations** (∀,∃,⊕,⊗), keyed on the **monad `m`** because the aggregation IS the Kleisli bind of `m` (the point type is a method-level `forall a`, not a class index). `Signature.Guard m a` = the data a quantifier ranges over: `[a]` for `Dist` (a list to fold), the batched tensor `a` for `LogTens` (one vectorized op). The truth object is `Bool`. Interpretations: `Interpretations.Boolean` (the crisp universe-free `instance TwoMonBLat Bool` + `instance A2MonBLat Dist Bool`, the `mapM`-then-fold quantifier) and `Interpretations.TensorBool` (`instance A2MonBLat LogTens Bool`, the batched logsumexp quantifier, plus the `logTensNLL` / `logTensPTrue` readouts over the leaf logits).
  - `C_Domain.NeuralNets.*` (the shared C-layer machinery) — reusable **architectures-as-data**, split signature/interpretation. (Terminology: an *architecture* is the parameter-free structure `Arch`; a *model* = architecture + θ, realized only as `runArch arch θ`. The two are kept separate — there is no "Model" object — so what lives here are architectures, hence the naming.) `C_Domain.NeuralNets.DSL.Syntax` is the abstract vocabulary (`data Layer = Linear … | Conv2d … | ELU | ReLU | Sigmoid | MaxPool | Flatten`; `type Arch = [Layer]`, written as a list literal e.g. `[Linear 2 16, ELU, Linear 16 1]`, with `(>>>)` to glue reusable blocks). `C_Domain.NeuralNets.DSL.Semantics` gives it meaning: `Weights` = θ (the **pure** sampled parameters — a `[LayerWeight]`, `Parameterized`, no forward/architecture inside), `sampleWeights :: Arch -> IO Weights` (draw θ₀) and `runArch :: Arch -> Weights -> Tensor -> Tensor` (the forward) — two interpretations of the same pure `Arch`, both **thin dispatchers**: each symbol's actual function lives in `C_Domain.NeuralNets.DSL.Library.*`, one file per layer-op grouped by kind (`Activation/` · `Parameterized/` · `Shape/` — the P=ℝ⁰ vs P≠0 split), with no `Torch.*` inlined in the dispatcher. `C_Domain.NeuralNets.{MLP,MnistCNN}` are named architectures, each exporting its `Arch` (`mlpArch`/`cnnArch`) **and its forward at θ** (`mlp`/`cnn`, defined `= runArch <arch>`), so a call site writes `cnn θ x` instead of `runArch cnnArch θ x`. To add a learnable layer: a `Layer` symbol + a `LayerWeight` constructor + a `C_Domain.NeuralNets.DSL.Library.*` function + a case in `sampleWeights`/`runArch`.
  - `F_Inferential.*` — the generic `train` (Adam loop), `InferenceSignature` + interpretation, the loss `Library` (`Softplus`, `NegLog`, `CrossEntropy`, `OneMinus`, `Convex`).
  - `G_Statistical.*` — the flexible `Report` (labeled metrics) + `printReport`/`averageReports`/`runAverage`, `BenchmarkSignature` + interpretation, the metric `Library`.
  - `Example` — the `Example` typeclass + `runExample` (train + benchmark). `Run` — the `nesycat` dispatcher.
  - (There is no shared `E_Data`: data is inherently per-example, so the E layer lives only inside each example.)

- **`Examples/<Name>/`** — one self-contained example = the **C–G stack of folders** (plus `Definition.hs`). **A and B are LIBRARY-ONLY** — always reused, never overridden per example, so they are NOT example folders; they are wired in directly via `Definition.hs` imports (`A_Categorical.Monads.LogTens`, `B_Logical.Interpretations.{Boolean,TensorBool}`). Each present layer folder mirrors the Library's `Signature` / `Interpretation` split. Data (E) comes **before** the inferential layer (F): the objective needs both the axiom (D) and the data, so both precede F.

  | layer folder | role (what it must provide) |
  |---|---|
  | `C_Domain/{Signature,Interpretation}.hs` | C — the domain's **sorts** (monad-INVARIANT plain types, e.g. `Image`/`Digit`/`Natural`/`Point`/`Omega = Bool`) + the **neural Kleisli symbols** (`digit` — a Kleisli *function* `Image → m Digit`, class `MnistKlFun`; `classifierA`/`labelA` — Kleisli *relations* into `Omega = Bool`, class `BinaryKlRel`; a class over `m`), interpreted per monad (`@Dist`/`@LogTens`); the parameter space is exposed as `Params = Weights` + `initParams` (draw θ₀) |
  | `D_Grammatical/{Signature,Interpretation}.hs` | D — the axiom: one abstract formula `forall m` (`Signature`) + BOTH readings side by side in one `Interpretation.hs` (the `@Dist` probability reading and the `@LogTens` training reading, the `sat`) |
  | `E_Data/{Signature,Loader}.hs` | E — the data **format** (`Signature`: the `Dataset` record) + the **loader** (`Loader`: `loadData`); data files committed beside it (e.g. `Examples/MnistAddition/E_Data/`) |
  | `F_Inferential/Interpretation.hs` | F — `objective` (the inference penalty of D's axiom over the E data) + `trainConfig`; signature reused |
  | `G_Statistical/Interpretation.hs` | G — this example's metrics as a labeled `Report`; signature reused |
  | `Definition.hs` | the A–G **manifest** (see below) |

  **`Definition.hs`** is the manifest: an empty `data <Name>` + `instance Example` wiring each member to its layer (`initParams`←C, `loadData`/`batches`←E, `sat`←D, `trainConfig`←F, `report`←G; the name is the folder name, from the dispatcher). The objective itself is GENERIC and lives in `runExample` (`lossKnow . sat`); F only supplies `trainConfig` and (via a shared `instance InferenceSignature (Truth e)`) the loss `lossKnow`. For **each layer slot it is exactly one of two things** — a **reuse** of an already-made shared module (imported directly in `Definition.hs`: A and B always, plus the F/G *signatures*), or this example's **own standalone file** (in the layer folder). No forwarding wrappers. `Signature` and `Interpretation` are independent slots, so e.g. F/G reuse the shared `InferenceSignature`/`BenchmarkSignature` but supply their own interpretation. Named `Definition` (not `Example`) to avoid colliding with the Library's `Example` class. A and B are always reused this way (a library import in `Definition.hs`, no example folder); `Binary`, `MnistAddition`, `MnistMultiDigit` also reuse the F/G signatures; everything else is standalone.

- The single executable's `Main.hs` is a 3-line shim at the repo root (dispatch logic is in `Run`); there is no `app/` directory.

## Adding a new example (the scaffolding "button")

```bash
Examples/new-example.sh SudokuSolver   # UpperCamelCase (just scaffolds the folder)
./nesycat SudokuSolver 1               # auto-registered by folder name; builds + runs the stub
```

`new-example.sh` copies `Examples/_template/` (a C–G stack of compilable stubs, plus `Definition.hs`) and renames the `Template` placeholder — that's all. There is **no registration step**: `./nesycat` discovers the new `Examples/SudokuSolver/` folder, regenerates `Library/Run.hs`, re-globs via hpack, and runs it. Then fill in the standalone slots: `C_Domain/*`, `D_Grammatical/*`, `E_Data/*`, `F_Inferential/Interpretation.hs`, `G_Statistical/Interpretation.hs` (A/B are reused purely via `Definition.hs` library imports — no example folder).

> **Caveat:** `Examples/_template/` has **not yet been migrated** to the current monad-`m` style — it still carries the old `GeomU`/`MeasU` universe scaffolding and the split `D_Grammatical/{InterpretationData,InterpretationTens}`. It is **not built** (hpack skips leading-underscore dirs), so it doesn't break anything, but a freshly-scaffolded example currently needs hand-cleanup to the monad-`m` form (parametrize over `m`, `@Dist`/`@LogTens`, merge D into one `Interpretation.hs`) until the template is refreshed.

## Key patterns

- **One formula, two interpretations.** A formula is written once (abstract over the monad `m`, with `plus`/`eqNat` as plain host ops and the binds doing the work) and read at **`@LogTens`** (log-space measures, raw logits, logsumexp — the differentiable *training* reading, the `sat`) and **`@Dist`** (probabilities — the law of total probability via the Kleisli bind, the *probability reading*). Only the symbol interpretations change, never the do-block. **The observation is data that enters the monad** (`◯` in the theory): it is a CERTAIN monadic value `n :: m Natural` (= `η n`), **bound `s <- n` exactly like the digits** — so `(+)`/`(=)` are plain host ops on the three bound values (`do { d1 <- dig x; d2 <- dig y; s <- n; return (s .= d1 .+ d2) }`). In `Dist` that's `pure n`; in `LogTens` it's a batched one-hot leaf (`encode` = `η` realized for a batch). That leaf is **built by the E/data layer** (the batch already carries `LogTens Natural`), so the D interpretation is a pure pass-through (`mnistSentence @LogTens () batch theta`) — no lifting/encoding in the interpretation.
- **The objective only touches the grammatical axiom over data** — never the model directly. It reaches the net only through the interpretation (`digit @LogTens`, `classifierA @LogTens`): the generic objective is `lossKnow . sat`, `sat` = the `@LogTens` reading of the axiom over a batch.
- **`LogTens` stays in logit space**; softmax → probabilities happens only at the `decode` bridge (the `@Dist` reading of a prediction, `digit @Dist = decode . digit @LogTens`) or inside the loss readout (`logTensNLL = logDen − logNum`, pure logsumexp — no `exp`/softmax/clamp on the training path). The marginalization **emerges from the bind as a log-space convolution** (variable elimination): `logConvolve` (`A_Categorical.Monads.LogTensExpect`) folds the formula's independent leaves, merging equal partial sums in log space — no `O(∏ kᵢ)` joint (so e.g. multi-digit's would-be `[B,10,10,10,10,199] ≈ 0.5 GB` joint never forms). `logNumDen` (`B_Logical.Interpretations.TensorBool`) routes any *separable*-observation predicate through it — `obs == base + Σᵢ cᵢ(xᵢ)`, the `cᵢ` **discovered by probing** (sums, weighted sums, counts, Binary's iff; NOT hardcoded to `+`) — and falls back to the full-joint `marginalize` (kept as oracle/fallback) for non-separable predicates. (Efficient marginalization of an arbitrary predicate is impossible — the treewidth/no-free-lunch bound; knowledge compilation is the general route, a future engine.) `mapLeafWeights` (same module) slices/gathers an observation leaf along the batch dim, so the E layer mini-batches `η n` without leaving the monad.
- **The report is one flexible labeled-metrics type** (`G_Statistical.Report`); each example reports its own honestly-named metrics (no field-cramming).
- **Sorts are monad-INVARIANT plain types** (`Image`/`Digit`/`Natural`/`Point`/`Omega = Bool`) — there is no per-monad sort assignment and no representation bridge for them (an image is the same tensor in both readings). The ONLY monad-dependent symbols are the **neural Kleisli symbols** — `digit` is a Kleisli *function* (`Image → m Digit`, lands in a sort; class `MnistKlFun`), while `classifierA`/`labelA` are Kleisli *relations* (land in `Omega = Bool`; class `BinaryKlRel`) — a class over `m` whose monad is `Dist` vs `LogTens`. The parameter space is the **pure parameters** `Weights` (θ), exposed as `Params = Weights`; the C *interpretation* picks the architecture by **importing** it and reaches the net via its named forward (`mlp θ` / `cnn θ`), with `initParams = sampleWeights mlpArch` (`cnnArch`) drawing θ₀. These symbols take **just θ** — never an architecture or a forward — and `Randomizable`/`sample` stays hidden inside `sampleWeights` (no `Spec` in the contract).
- Untyped `Torch.*` tensors throughout (`Torch.Tensor`), with `@LogTens`/`@Dist` type applications selecting the monad (the reading); type families + type classes give the signature/interpretation separation at every layer.

## Conventions

- Keep the by-example structure: a new example is a self-contained `Examples/<Name>/` folder (the C–G stack + `Definition.hs` + its own `data/`; A/B are library-only, imported in `Definition.hs`); shared/reusable code goes in `Library/`.
- After changing the module set, run `hpack --force` and commit the regenerated `.cabal`.
