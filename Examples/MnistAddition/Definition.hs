{-# LANGUAGE TypeFamilies #-}

-- | MnistAddition example — the A–G manifest. Each layer slot is filled in exactly
--   ONE of two ways: a REUSE of an already-made template (a shared module,
--   referenced directly here), or this example's OWN STANDALONE file (in its layer
--   folder). Signature and Interpretation are independent slots, so F and G reuse
--   the template signature while supplying their own interpretation.
module MnistAddition.Definition (MnistAddition) where

-- A (category) — REUSE shared: the LogVec monad (the truth object's carrier).
import A_Categorical.Category.Monads.LogVec (LogVec)
-- B (logic) — REUSE template: the library Boolean (the shared crisp-Bool truth algebra +
--   the Dist quantifier) + TensorBool (only the LogVec quantifier for Bool).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.TensorBool ()
-- C (domain) — STANDALONE: MNIST's own sorts/symbols + model (Params/initParams).
import qualified MnistAddition.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: the axiom in both monads (one file); 'sat' = the LogVec reading.
import qualified MnistAddition.D_Grammatical.Interpretation as D
-- E (data) — STANDALONE: MNIST's own data format + loader (exports the data + the batches).
import qualified MnistAddition.E_Data.Signature as E
import qualified MnistAddition.E_Data.Loader as EL
-- F (inference) — REUSE template signature AND the library's shared
--   @instance InferenceSignature (LogVec Bool)@ (negLog . logVecPTrue); STANDALONE only for trainConfig.
import F_Inferential.InferenceSignature ()
import F_Inferential.InferenceInterpretation ()
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
  type Truth MnistAddition = LogVec Bool
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  batches = EL.batches
  sat = D.mnistAxiomTens
  report = G.report
