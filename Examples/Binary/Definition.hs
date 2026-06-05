{-# LANGUAGE TypeFamilies #-}

-- | Binary example — the A–G manifest. Each layer slot is filled in exactly ONE of two
--   ways: a REUSE of an already-made template (a shared module, referenced directly
--   here), or this example's OWN STANDALONE file. The inference layer (F) is just the
--   loss choice: Binary REUSES the library's logit-truth instance, so its own F only
--   carries 'trainConfig'; D exports the satisfaction 'sat', E the data + batches.
module Binary.Definition (Binary) where

-- A (category) — REUSE shared: the LogVec monad (the truth object's carrier).
import A_Categorical.Category.Monads.LogVec (LogVec)
-- B (logic) — REUSE template: the library Boolean (the shared crisp-Bool truth algebra +
--   the Dist quantifier) + TensorBool (the LogVec quantifier for Bool).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.TensorBool ()
-- C (domain) — STANDALONE: Binary's own sorts/symbols + parameter space (Params/initParams).
import qualified Binary.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: the axiom in both monads (one file); 'sat' = the LogVec reading.
import qualified Binary.D_Grammatical.Interpretation as D
-- E (data) — STANDALONE: Binary's own data format + loader (exports the data + the batches).
import qualified Binary.E_Data.Signature as E
import qualified Binary.E_Data.Loader as EL
-- F (inference) — REUSE template signature AND the library's shared probabilistic-truth loss
--   (instance InferenceSignature (LogVec Bool) = negLog . logVecPTrue); STANDALONE only for trainConfig.
import F_Inferential.InferenceSignature ()
import F_Inferential.InferenceInterpretation ()
import qualified Binary.F_Inferential.Interpretation as F
-- G (statistics) — REUSE template signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified Binary.G_Statistical.Interpretation as G
import Example (Example (..))
import qualified Torch

data Binary

instance Example Binary where
  type Params Binary = C.Params
  type Data Binary = E.Dataset
  type Batch Binary = Torch.Tensor
  type Truth Binary = LogVec Bool
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  batches = EL.batches
  sat = D.binaryAxiomTens
  report = G.report
