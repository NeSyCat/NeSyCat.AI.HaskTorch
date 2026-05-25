---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Haskell implementation of the **NeSyCat** neurosymbolic framework using HaskTorch (untyped `Torch.*` bindings to libtorch). The code is a typed tensor-category realization of the theory in the companion paper repo (`../NeSyCat Papers/`).

## Build & Run Commands

The build is driven by **hpack**: `package.yaml` is the source of truth and generates `nesycat-hasktorch.cabal` (committed). Re-run hpack after adding/removing/renaming modules.

```bash
~/.cabal/bin/hpack --force            # regenerate the .cabal from package.yaml (after module changes)
cabal build all                       # build the library + the single `nesycat` executable

# Run an example:  nesycat <name> [n]   (n = runs to average; n=1 prints the loss curve)
cabal run nesycat -- binary 1
cabal run nesycat -- mnist-add 1      # data ships in the example; Examples/MnistAddition/G_Data/get-mnist.sh refetches it
cabal run nesycat -- binary +RTS -s   # RTS stats (exe built with -rtsopts)
```

Requires `hasktorch` + libtorch. HLS via `hie.yaml` (cabal cradle, reads the generated `.cabal`). `hpack` lives at `~/.cabal/bin/hpack`.

## Architecture: by-example over a shared library

The repo is organized **by example**, not by layer. Two folders at the repo root hold everything — `Library/` (shared) and `Examples/` (per example):

- **`Library/`** — the shared, reusable framework, keeping the A–G layer names:
  - `A_Categorical.*` — universes `GeomU`/`MeasU`, monads (`Ident`/`Dist`/`Giry`) + `eta`/`mu`, the `Universe` class.
  - `B_Logical.*` — `LogicalSignature` / `LogicalQuantSignature` (∨,∧,⊕,⊗,∀,∃) and the interpretations `Boolean` (MeasU) and `Tensor`/TensReal (GeomU; LogSumExp on logits).
  - `E_Inferential.*` — the generic `train` (Adam loop), `InferenceSignature` + interpretation, the loss `Library` (`Softplus`, `NegLog`, `CrossEntropy`, `OneMinus`, `Convex`).
  - `F_Statistical.*` — the flexible `Report` (labeled metrics) + `printReport`/`averageReports`/`runAverage`, `BenchmarkSignature` + interpretation, the metric `Library`.
  - `C_Domain.Models.*` — reusable domain models (`MLP`, `MnistCNN`): the C-layer neural functions an example plugs into its domain interpretation.
  - `Example` — the `Example` typeclass + `runExample` (train + benchmark). `Run` — the `nesycat` dispatcher.

- **`Examples/<Name>/`** — one self-contained example = the full A–G stack:

  | module | layer | role |
  |---|---|---|
  | `A_Categorical.hs` | α | which universe(s) — usually re-exports `A_Categorical` |
  | `B_Logical.hs` | β | which logic — usually re-exports `B_Logical` |
  | `C_Domain/{Signature,Interpretation}.hs` | γ | domain sorts + symbols, interpreted per universe |
  | `D_Grammatical/{Formulas,InterpretationData,InterpretationTens}.hs` | δ | the axiom (one abstract formula) + its MeasU(`Dist`) and GeomU(tensor) interpretations |
  | `E_Inferential.hs` | ε | the inference interpretation (the penalty) |
  | `F_Statistical.hs` | ζ | this example's metrics, as a labeled `Report` |
  | `G_Data/` | — | the loader (`Loader.hs`) **and** this example's data files, together (committed), e.g. `Examples/MnistAddition/G_Data/` |
  | `Example.hs` | — | the `Example` instance wiring A–G together |

  Each layer slot can **reuse** the library (re-export), **modify** it, or be filled from the template. Example-specific data lives in that example's own `G_Data/` folder, beside its `Loader.hs`. Existing examples: `Binary` (circle-in-square classification) and `MnistAddition` (single-digit addition; digits learned from observed sums alone).

- The single executable's `Main.hs` is a 3-line shim at the repo root (dispatch logic is in `Run`); there is no `app/` directory.

## Adding a new example (the scaffolding "button")

```bash
Examples/new-example.sh SudokuSolver  # UpperCamelCase
cabal run nesycat -- sudoku-solver 1      # builds + runs the stub immediately
```

`new-example.sh` copies `Examples/_template/` (a full A–G stack of compilable stubs), renames the `Template` placeholder, registers the example in `Library/Run.hs` (between its `NEW-EXAMPLE-*` markers), and re-runs hpack — so the new modules build with **no manual cabal edit**. Then fill in the `C_Domain`, `D_Grammatical`, `G_Data` slots (and tweak `E_Inferential`/`F_Statistical`).

## Key patterns

- **One formula, two interpretations.** A formula is written once (abstract over the universe `u`) and interpreted in **GeomU** (`Identity` monad, tensors/logits, TensReal/LogSumExp — used for differentiable *training*) and **MeasU** (`Dist` monad, probabilities — the law of total probability via the Kleisli bind, used for the *probability reading*). Only the symbol interpretations change, never the formula.
- **The objective only touches the grammatical axiom over data** — never the model directly. It reaches the net only through the interpretation (`classifierA @GeomU`, `digit @GeomU`): `objective = <inference penalty> (axiom β data θ)`.
- **GeomU stays in logit space**; softmax/sigmoid → probabilities happens only at the MeasU bridge (`decOmega`/`decDigit`) or in the inference penalty (e.g. MNIST's categorical NLL `mnistKnowLoss`).
- **The report is one flexible labeled-metrics type** (`F_Statistical.Report`); each example reports its own honestly-named metrics (no field-cramming).
- Untyped `Torch.*` tensors throughout (`Torch.Tensor`), with `@GeomU`/`@MeasU` type applications selecting the universe; type families + type classes give the signature/interpretation separation at every layer.

## Conventions

- Keep the by-example structure: a new example is a self-contained `Examples/<Name>/` folder (full A–G stack + its own `data/`); shared/reusable code goes in `Library/`.
- After changing the module set, run `hpack --force` and commit the regenerated `.cabal`.
