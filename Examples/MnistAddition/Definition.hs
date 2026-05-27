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
-- B (logic) — REUSE template: the library Boolean (MeasU) + TensorProb (GeomU, the
--   [0,1] product fuzzy logic; it re-exports the Tensor Guard instance it builds on).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.TensorProb (OmegaP)
-- C (domain) — STANDALONE: MNIST's own sorts/symbols + model (Params/initParams).
import qualified MnistAddition.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: builds the satisfaction 'sat' (= conjunction of the axioms).
import qualified MnistAddition.D_Grammatical.InterpretationTens as D
-- E (data) — STANDALONE: MNIST's own data format + loader (exports the data + the batches).
import qualified MnistAddition.E_Data.Signature as E
import qualified MnistAddition.E_Data.Loader as EL
-- F (inference) — REUSE template signature; STANDALONE interpretation = ONLY the loss
--   choices (instance InferenceSignature OmegaP) + trainConfig. Imported for its instance.
import F_Inferential.InferenceSignature ()
import qualified MnistAddition.F_Inferential.Interpretation as F
-- G (statistics) — REUSE template signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified MnistAddition.G_Statistical.Interpretation as G
import Example (Example (..))
import qualified Torch

data MnistAddition

instance Example MnistAddition where
  type Params MnistAddition = C.Params
  type Data MnistAddition = E.Dataset
  type Batch MnistAddition = (Torch.Tensor, Torch.Tensor, Torch.Tensor)
  type Truth MnistAddition = OmegaP
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  batches = EL.batches
  sat = D.mnistSat
  report = G.report
