{-# LANGUAGE TypeFamilies #-}

-- | MnistAddition example — the A–G manifest. Each layer slot is filled in exactly
--   ONE of two ways: a REUSE of an already-made template (a shared module,
--   referenced directly here), or this example's OWN STANDALONE file (in its layer
--   folder). Signature and Interpretation are independent slots, so F and G reuse
--   the template signature while supplying their own interpretation.
module MnistAddition.Definition (MnistAddition) where

-- A (category) — REUSE template: the library universes + categorical signature.
import A_Categorical.CategoricalSignature ()
import A_Categorical.CategoricalInterpretation ()
-- B (logic) — REUSE template: the library Boolean (MeasU) + Tensor (GeomU).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.Tensor ()
-- C (domain) — STANDALONE: MNIST's own sorts/symbols + model (Params/Spec/spec).
import qualified MnistAddition.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: the GeomU sum-logits term (consumed by F.objective).
import MnistAddition.D_Grammatical.InterpretationTens ()
-- E (data) — STANDALONE: MNIST's own data format + loader.
import qualified MnistAddition.E_Data.Signature as E
import qualified MnistAddition.E_Data.Loader as EL
-- F (inference) — REUSE template signature; STANDALONE interpretation (categorical NLL).
import F_Inferential.InferenceSignature ()
import qualified MnistAddition.F_Inferential.Interpretation as F
-- G (statistics) — REUSE template signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified MnistAddition.G_Statistical.Interpretation as G
import Example (Example (..))

data MnistAddition

instance Example MnistAddition where
  type Params MnistAddition = C.Params
  type Data MnistAddition = E.Dataset
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  objective = F.objective
  report = G.report
