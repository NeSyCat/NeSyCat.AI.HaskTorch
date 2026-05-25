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
  - `C_Domain.Models.*` — reusable domain models (`MLP`, `MnistCNN`): the C-layer neural functions an example plugs into its domain interpretation.
  - `F_Inferential.*` — the generic `train` (Adam loop), `InferenceSignature` + interpretation, the loss `Library` (`Softplus`, `NegLog`, `CrossEntropy`, `OneMinus`, `Convex`).
  - `G_Statistical.*` — the flexible `Report` (labeled metrics) + `printReport`/`averageReports`/`runAverage`, `BenchmarkSignature` + interpretation, the metric `Library`.
  - `Example` — the `Example` typeclass + `runExample` (train + benchmark). `Run` — the `nesycat` dispatcher.
  - (There is no shared `E_Data`: data is inherently per-example, so the E layer lives only inside each example.)

- **`Examples/<Name>/`** — one self-contained example = the full A–G stack. Data (E) comes **before** the inferential layer (F): the objective needs both the axiom (D) and the data, so both precede F.

  | module | role (what it must provide) |
  |---|---|
  | `A_Categorical.hs` | A — which universe(s); usually re-exports `A_Categorical` |
  | `B_Logical.hs` | B — which logic; usually re-exports `B_Logical` |
  | `C_Domain/{Signature,Interpretation}.hs` | C — domain sorts + symbols (per universe) **and** the model it uses, exposed as `Params`/`Spec`/`spec` |
  | `D_Grammatical/{Formulas,InterpretationData,InterpretationTens}.hs` | D — the axiom (one abstract formula) + its MeasU(`Dist`) and GeomU(tensor) interpretations |
  | `E_Data/Loader.hs` | E — the dataset (`Dataset`/`loadData`), in the fixed tensor format, prepared independently of training; data files are committed beside it (e.g. `Examples/MnistAddition/E_Data/`) |
  | `F_Inferential.hs` | F — `objective` (the inference penalty of D's axiom over the E data) + `trainConfig` |
  | `G_Statistical.hs` | G — this example's metrics, as a labeled `Report` |
  | `Example.hs` | the A–G **manifest**: an empty `data <Name>` + `instance Example` that only points each member at its layer (`spec`←C, `loadData`←E, `objective`/`trainConfig`←F, `report`←G). The name is the folder name (supplied by the dispatcher), so it isn't written here. |

  Each layer slot can **reuse** the library (re-export), **modify** it, or be filled from the template. Example-specific data lives in that example's own `E_Data/` folder, beside its `Loader.hs`. Existing examples: `Binary` (circle-in-square classification) and `MnistAddition` (single-digit addition; digits learned from observed sums alone).

- The single executable's `Main.hs` is a 3-line shim at the repo root (dispatch logic is in `Run`); there is no `app/` directory.

## Adding a new example (the scaffolding "button")

```bash
Examples/new-example.sh SudokuSolver   # UpperCamelCase (just scaffolds the folder)
./nesycat SudokuSolver 1               # auto-registered by folder name; builds + runs the stub
```

`new-example.sh` copies `Examples/_template/` (a full A–G stack of compilable stubs) and renames the `Template` placeholder — that's all. There is **no registration step**: `./nesycat` discovers the new `Examples/SudokuSolver/` folder, regenerates `Library/Run.hs`, re-globs via hpack, and runs it. Then fill in the `C_Domain`, `D_Grammatical`, `E_Data/Loader.hs` slots (and tweak `F_Inferential`/`G_Statistical`).

## Key patterns

- **One formula, two interpretations.** A formula is written once (abstract over the universe `u`) and interpreted in **GeomU** (`Identity` monad, tensors/logits, TensReal/LogSumExp — used for differentiable *training*) and **MeasU** (`Dist` monad, probabilities — the law of total probability via the Kleisli bind, used for the *probability reading*). Only the symbol interpretations change, never the formula.
- **The objective only touches the grammatical axiom over data** — never the model directly. It reaches the net only through the interpretation (`classifierA @GeomU`, `digit @GeomU`): `objective = <inference penalty> (axiom β data θ)`.
- **GeomU stays in logit space**; softmax/sigmoid → probabilities happens only at the MeasU bridge (`decOmega`/`decDigit`) or in the inference penalty (e.g. MNIST's categorical NLL `mnistKnowLoss`).
- **The report is one flexible labeled-metrics type** (`G_Statistical.Report`); each example reports its own honestly-named metrics (no field-cramming).
- Untyped `Torch.*` tensors throughout (`Torch.Tensor`), with `@GeomU`/`@MeasU` type applications selecting the universe; type families + type classes give the signature/interpretation separation at every layer.

## Conventions

- Keep the by-example structure: a new example is a self-contained `Examples/<Name>/` folder (full A–G stack + its own `data/`); shared/reusable code goes in `Library/`.
- After changing the module set, run `hpack --force` and commit the regenerated `.cabal`.
