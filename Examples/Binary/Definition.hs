{-# LANGUAGE TypeFamilies #-}

-- | Binary example — the A–G manifest. Each layer slot is filled in exactly ONE of
--   two ways: a REUSE of an already-made template (a shared module, referenced
--   directly here), or this example's OWN STANDALONE file (in its layer folder).
--   Signature and Interpretation are independent slots, so a layer can reuse the
--   template signature while supplying its own interpretation — as F and G do.
module Binary.Definition (Binary) where

-- A (category) — REUSE template: the library universes + categorical signature.
import A_Categorical.CategoricalSignature ()
import A_Categorical.CategoricalInterpretation ()
-- B (logic) — REUSE template: the library Boolean (MeasU) + Tensor (GeomU).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.Tensor ()
-- C (domain) — STANDALONE: Binary's own sorts/symbols + model (Params/Spec/spec).
import qualified Binary.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: Binary's own axiom (consumed by F.objective).
import Binary.D_Grammatical.InterpretationTens ()
-- E (data) — STANDALONE: Binary's own data format + loader.
import qualified Binary.E_Data.Signature as E
import qualified Binary.E_Data.Loader as EL
-- F (inference) — REUSE template signature; STANDALONE interpretation (objective + trainConfig).
import F_Inferential.InferenceSignature ()
import qualified Binary.F_Inferential.Interpretation as F
-- G (statistics) — REUSE template signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified Binary.G_Statistical.Interpretation as G
import Example (Example (..))

data Binary

instance Example Binary where
  type Params Binary = C.Params
  type Spec Binary = C.Spec
  type Data Binary = E.Dataset
  spec = C.spec
  loadData = EL.loadData
  trainConfig = F.trainConfig
  objective = F.objective
  report = G.report
