{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

-- | The library's default inference interpretations -- the loss choices for the two
--   standard truth objects, reused by examples rather than re-declared per example:
--
--     * @Torch.Tensor@ (a LOGIT satisfaction, the TensReal/Real-Logic reading):
--         lossKnow = softplus,  lossData = crossEntropy,  lossComb = convex.
--     * @LogTens Bool@ (a PROBABILISTIC satisfaction, the log-space sibling of @Dist Bool@):
--         lossKnow = 'negLogSat' (the mean @logDen - logNum@ off the 'logNumDen' marginal, in log
--         space -- so @J = -log SAT@; for the product @bigWedge@ this is the NLL / binary
--         cross-entropy). No data loss: lossData/lossComb stay at the class default (this reading's
--         objective is purely @lossKnow . sat@, so no probability is ever materialized).
--
--   Both MnistAddition and Binary now satisfy in @LogTens Bool@, so they share the second
--   instance and their F layers carry only @trainConfig@. An example with a genuinely
--   different truth object declares its own instance in its F layer.
module F_Inferential.InferenceInterpretation () where

import A_Categorical.Monads.LogTens (LogTens)
import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Library.Convex (convex)
import F_Inferential.Library.CrossEntropy (crossEntropy)
import F_Inferential.Library.NegLogSat (negLogSat)
import F_Inferential.Library.Softplus (softplus)
import qualified Torch

-- | Logit truth (Real-Logic reading): penalize a negative satisfaction logit with softplus.
instance InferenceSignature Torch.Tensor where
  type Loss Torch.Tensor = Torch.Tensor
  lossKnow = softplus
  lossData = crossEntropy
  lossComb = convex

-- | Probabilistic truth (the @LogTens Bool@ sentence): the knowledge loss is 'negLogSat', the
--   negative-log satisfaction read DIRECTLY in log space (@logDen - logNum@ off 'logNumDen') --
--   training never forms a probability (no @exp@, no clamp). This reading is purely
--   knowledge-driven (@lossKnow . sat@), so it provides NO data loss: lossData/lossComb stay at the
--   class default. Shared by every example whose satisfaction is a @LogTens Bool@ (MnistAddition, Binary).
instance InferenceSignature (LogTens Bool) where
  type Loss (LogTens Bool) = Torch.Tensor
  lossKnow = negLogSat
