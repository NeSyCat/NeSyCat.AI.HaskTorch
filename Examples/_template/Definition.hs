{-# LANGUAGE TypeFamilies #-}

-- | Template example — the A–G manifest. Each layer slot is filled in exactly ONE
--   of two ways: a REUSE of an already-made shared module (referenced directly
--   here), or this example's OWN STANDALONE file (in its layer folder). Reused
--   layers keep an empty folder as a ready-to-fill slot. This stub builds and runs
--   as-is (@nesycat <name> 1@); fill in the C/D/E/F/G layers to make it real.
module Template.Definition (Template) where

-- A (category) — REUSE shared: the library universes + categorical signature.
import A_Categorical.CategoricalSignature ()
import A_Categorical.CategoricalInterpretation ()
-- B (logic) — REUSE shared: the library Boolean (MeasU) + Tensor (GeomU).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.Tensor ()
-- C (domain) — STANDALONE: this example's own sorts/symbols + parameter space (Params/initParams).
import qualified Template.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: this example's own axiom.
import Template.D_Grammatical.InterpretationTens ()
-- E (data) — STANDALONE: this example's own data format + loader.
import qualified Template.E_Data.Signature as E
import qualified Template.E_Data.Loader as EL
-- F (inference) — REUSE shared signature; STANDALONE interpretation (objective + trainConfig).
import F_Inferential.InferenceSignature ()
import qualified Template.F_Inferential.Interpretation as F
-- G (statistics) — REUSE shared signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified Template.G_Statistical.Interpretation as G
import Example (Example (..))

data Template

instance Example Template where
  type Params Template = C.Params
  type Data Template = E.Dataset
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  objective = F.objective
  report = G.report
