-- | Negative-log satisfaction knowledge loss for the @LogTens Bool@ truth object:
--   @pen(m) = mean (logDen - logNum)@, read DIRECTLY off the log-space marginal
--   'B_Logical.Interpretations.TensorBool.logNumDen' -- pure 'logsumexp' arithmetic, no
--   @exp@-to-probability and no clamp, so the whole training path stays in logits with a full
--   gradient. For the product @bigWedge@ this is the NLL / binary cross-entropy of the axiom. The
--   @LogTens Bool@ analogue of 'F_Inferential.Library.Softplus.softplus' (the logit reading's
--   knowledge loss).
module F_Inferential.Library.NegLogSat (negLogSat) where

import A_Categorical.Monads.LogTens (LogTens)
import B_Logical.Interpretations.TensorBool (logNumDen)
import qualified Torch

negLogSat :: LogTens Bool -> Torch.Tensor
negLogSat m = let (logNum, logDen) = logNumDen m in Torch.mean (logDen `Torch.sub` logNum)
