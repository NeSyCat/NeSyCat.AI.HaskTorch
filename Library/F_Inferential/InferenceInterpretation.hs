{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

-- | The library's default inference interpretations -- the loss choices for the two
--   standard truth objects, reused by examples rather than re-declared per example:
--
--     * @Torch.Tensor@ (a LOGIT satisfaction, the TensReal/Real-Logic reading):
--         lossKnow = softplus,  lossData = crossEntropy,  lossComb = convex.
--     * @LogVec Bool@ (a PROBABILISTIC satisfaction, the log-space sibling of @Dist Bool@):
--         read out to a [0,1] degree with 'logVecPTrue' (the twin of @distPTrue@), then
--         lossKnow = negLog (so @J = -log SAT@; for the product @bigWedge@ this is the NLL /
--         binary cross-entropy), lossData = crossEntropy, lossComb = convex.
--
--   Both MnistAddition and Binary now satisfy in @LogVec Bool@, so they share the second
--   instance and their F layers carry only @trainConfig@. An example with a genuinely
--   different truth object declares its own instance in its F layer.
module F_Inferential.InferenceInterpretation () where

import A_Categorical.Category.Monads.LogVec (LogVec)
import B_Logical.Interpretations.TensorBool (logVecNLL, logVecPTrue)
import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Library.Convex (convex)
import F_Inferential.Library.CrossEntropy (crossEntropy)
import F_Inferential.Library.Softplus (softplus)
import qualified Torch

-- | Logit truth (Real-Logic reading): penalize a negative satisfaction logit with softplus.
instance InferenceSignature Torch.Tensor where
  type Loss Torch.Tensor = Torch.Tensor
  lossKnow = softplus
  lossData = crossEntropy
  lossComb = convex

-- | Probabilistic truth (the @LogVec Bool@ sentence): the knowledge loss is the negative-log
--   satisfaction read DIRECTLY in log space ('logVecNLL' @= logDen - logNum@) -- training never
--   forms a probability (no @exp@, no clamp). The data loss is a cross-entropy against the
--   probability reading ('logVecPTrue'). Shared by every example whose satisfaction is a
--   @LogVec Bool@ (MnistAddition, Binary).
instance InferenceSignature (LogVec Bool) where
  type Loss (LogVec Bool) = Torch.Tensor
  lossKnow m = Torch.mean (logVecNLL m)
  lossData m y = crossEntropy (logVecPTrue m) (logVecPTrue y)
  lossComb dataLoss knowLoss m = convex dataLoss knowLoss (logVecPTrue m)
