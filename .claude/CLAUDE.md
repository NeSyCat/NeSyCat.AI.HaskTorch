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
  - `A_Categorical.*` — universes `GeomU`/`MeasU`, monads (`Ident`/`Dist`/`Giry`) + `eta`/`mu`, the `Universe` class.
  - `B_Logical.*` — `LogicalSignature` / `LogicalQuantSignature` (∨,∧,⊕,⊗,∀,∃) and the interpretations `Boolean` (MeasU) and `Tensor`/TensReal (GeomU; LogSumExp on logits).
  - `C_Domain.Models.*` — reusable domain models as a shared **`Model` signature** (`C_Domain.Models.Signature`: `class Model space where forward :: space -> Tensor -> Tensor`) + **instances** (`C_Domain.Models.Interpretations.{MLP,MnistCNN}`): each network is a `Model` instance carrying its own parameter space + `forward` + `fresh` (its initializer, from an associated `Init` type), both as instance methods. Same class/instance (signature/interpretation) shape as `B_Logical`.
  - `F_Inferential.*` — the generic `train` (Adam loop), `InferenceSignature` + interpretation, the loss `Library` (`Softplus`, `NegLog`, `CrossEntropy`, `OneMinus`, `Convex`).
  - `G_Statistical.*` — the flexible `Report` (labeled metrics) + `printReport`/`averageReports`/`runAverage`, `BenchmarkSignature` + interpretation, the metric `Library`.
  - `Example` — the `Example` typeclass + `runExample` (train + benchmark). `Run` — the `nesycat` dispatcher.
  - (There is no shared `E_Data`: data is inherently per-example, so the E layer lives only inside each example.)

- **`Examples/<Name>/`** — one self-contained example = the full A–G stack, where **every layer A–G is a folder** (always present, even when empty), mirroring the Library's `Signature` / `Interpretation` split. Data (E) comes **before** the inferential layer (F): the objective needs both the axiom (D) and the data, so both precede F.

  | layer folder | role (what it must provide) |
  |---|---|
  | `A_Categorical/` | A — which universe(s); usually reused → **empty folder** (`.gitkeep`) |
  | `B_Logical/` | B — which logic; usually reused → **empty folder** |
  | `C_Domain/{Signature,Interpretation}.hs` | C — the domain's **vertical sorts** (data: `Point`/`Omega`, …) + **horizontal sorts** (parameter spaces, e.g. `Theta`) + relation symbols, interpreted per universe; the parameter space is exposed as `Params` + `initParams` (draw θ₀) |
  | `D_Grammatical/{Signature,InterpretationData,InterpretationTens}.hs` | D — the axiom (one abstract formula, in `Signature`) + its MeasU(`Dist`) and GeomU(tensor) interpretations |
  | `E_Data/{Signature,Loader}.hs` | E — the data **format** (`Signature`: the `Dataset` record) + the **loader** (`Loader`: `loadData`); data files committed beside it (e.g. `Examples/MnistAddition/E_Data/`) |
  | `F_Inferential/Interpretation.hs` | F — `objective` (the inference penalty of D's axiom over the E data) + `trainConfig`; signature reused |
  | `G_Statistical/Interpretation.hs` | G — this example's metrics as a labeled `Report`; signature reused |
  | `Definition.hs` | the A–G **manifest** (see below) |

  **`Definition.hs`** is the manifest: an empty `data <Name>` + `instance Example` wiring each member to its layer (`initParams`←C, `loadData`←E, `objective`/`trainConfig`←F, `report`←G; the name is the folder name, from the dispatcher). For **each layer slot it is exactly one of two things** — a **reuse** of an already-made template (a shared module, imported directly; that layer's folder then stays **empty** as a ready-to-fill placeholder), or this example's **own standalone file** (in the layer folder). No forwarding wrappers. `Signature` and `Interpretation` are independent slots, so e.g. F/G reuse the shared `InferenceSignature`/`BenchmarkSignature` but supply their own interpretation. Named `Definition` (not `Example`) to avoid colliding with the Library's `Example` class. Both `Binary` and `MnistAddition` reuse A/B + the F/G signatures (empty A/B folders); everything else is standalone.

- The single executable's `Main.hs` is a 3-line shim at the repo root (dispatch logic is in `Run`); there is no `app/` directory.

## Adding a new example (the scaffolding "button")

```bash
Examples/new-example.sh SudokuSolver   # UpperCamelCase (just scaffolds the folder)
./nesycat SudokuSolver 1               # auto-registered by folder name; builds + runs the stub
```

`new-example.sh` copies `Examples/_template/` (a full A–G stack of compilable stubs — all seven layer folders, A/B empty) and renames the `Template` placeholder — that's all. There is **no registration step**: `./nesycat` discovers the new `Examples/SudokuSolver/` folder, regenerates `Library/Run.hs`, re-globs via hpack, and runs it. Then fill in the standalone slots: `C_Domain/*`, `D_Grammatical/*`, `E_Data/*`, `F_Inferential/Interpretation.hs`, `G_Statistical/Interpretation.hs` (A/B are template references in `Definition.hs` — their folders are empty, ready to fill).

## Key patterns

- **One formula, two interpretations.** A formula is written once (abstract over the universe `u`) and interpreted in **GeomU** (`Identity` monad, tensors/logits, TensReal/LogSumExp — used for differentiable *training*) and **MeasU** (`Dist` monad, probabilities — the law of total probability via the Kleisli bind, used for the *probability reading*). Only the symbol interpretations change, never the formula.
- **The objective only touches the grammatical axiom over data** — never the model directly. It reaches the net only through the interpretation (`classifierA @GeomU`, `digit @GeomU`): `objective = <inference penalty> (axiom β data θ)`.
- **GeomU stays in logit space**; softmax/sigmoid → probabilities happens only at the MeasU bridge (`decOmega`/`decDigit`) or in the inference penalty (e.g. MNIST's categorical NLL `mnistKnowLoss`).
- **The report is one flexible labeled-metrics type** (`G_Statistical.Report`); each example reports its own honestly-named metrics (no field-cramming).
- **Two kinds of sorts (per domain `Signature`).** *Vertical sorts* = the data the formulas quantify over (`Point`/`Omega`; `Image`/`Digit`/`Natural`). *Horizontal sorts* = parameter spaces — the `Theta` where the learnable weights live (the "actor object"), in their own section/class (`BinaryParams`/`MnistParams`). The horizontal sort is realized by a **model** — a shared `Model` signature class (`forward`) with each network an instance under `C_Domain.Models.Interpretations.*` (or a bespoke `C_Domain/Model.hs`) — and exposed as `Params`; `initParams :: IO Params` draws θ₀ via the model's `fresh` method (e.g. `fresh (2, 16, 1)`). HaskTorch's `Randomizable`/`sample` stays hidden inside the model, so there is **no `Spec` in the contract**.
- Untyped `Torch.*` tensors throughout (`Torch.Tensor`), with `@GeomU`/`@MeasU` type applications selecting the universe; type families + type classes give the signature/interpretation separation at every layer.

## Conventions

- Keep the by-example structure: a new example is a self-contained `Examples/<Name>/` folder (full A–G stack + its own `data/`); shared/reusable code goes in `Library/`.
- After changing the module set, run `hpack --force` and commit the regenerated `.cabal`.
