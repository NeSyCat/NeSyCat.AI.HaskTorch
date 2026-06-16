{-# LANGUAGE TypeFamilies #-}

-- | MnistMultiDigit example — the A–G manifest. Two TWO-digit numbers added; only the sum is
--   observed. Each layer slot is a REUSE of shared library code or this example's own standalone
--   file. Reuses everything single-digit does (the LogTens monad, Boolean/TensorBool logic, the
--   MnistCNN, the generic train/loss/report) — only C/D/E/G are standalone, and the marginalization
--   over the four unknown digits is the shared log-space convolution (variable elimination).
module MnistMultiDigit.Definition (MnistMultiDigit) where

-- A (category) — REUSE shared: the LogTens monad (the truth object's carrier).
import A_Categorical.Monads.LogTens (LogTens)
-- B (logic) — REUSE template: the library Boolean (Dist quantifier) + TensorBool (LogTens quantifier).
import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.TensorBool ()
-- C (domain) — STANDALONE: the sorts/symbols + model (Params/initParams).
import qualified MnistMultiDigit.C_Domain.Interpretation as C
-- D (grammatical) — STANDALONE: the axiom in both monads (one file); 'sat' = the LogTens reading.
import qualified MnistMultiDigit.D_Grammatical.Interpretation as D
-- E (data) — STANDALONE: the data format + loader.
import qualified MnistMultiDigit.E_Data.Signature as E
import qualified MnistMultiDigit.E_Data.Loader as EL
-- F (inference) — REUSE the library's shared @instance InferenceSignature (LogTens Bool)@;
--   STANDALONE only for trainConfig.
import F_Inferential.InferenceSignature ()
import F_Inferential.InferenceInterpretation ()
import qualified MnistMultiDigit.F_Inferential.Interpretation as F
-- G (statistics) — REUSE template signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified MnistMultiDigit.G_Statistical.Interpretation as G
import Example (Example (..))
import qualified Torch

data MnistMultiDigit

instance Example MnistMultiDigit where
  type Params MnistMultiDigit = C.Params
  type Data MnistMultiDigit = E.Dataset
  type Batch MnistMultiDigit = (Torch.Tensor, Torch.Tensor, Torch.Tensor, Torch.Tensor, LogTens Int)
  type Truth MnistMultiDigit = LogTens Bool
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  batches = EL.batches
  sat = D.multiAxiomTens
  report = G.report
