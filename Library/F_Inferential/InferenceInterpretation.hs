{-# LANGUAGE TypeFamilies #-}

-- | TensReal inference interpretation: the loss choices for the LOGIT truth object
--   (@Torch.Tensor@), a library default that examples whose axiom reads in logit space
--   (e.g. Binary) reuse:
--
--     lossKnow |-> softplus   lossData |-> crossEntropy   lossComb |-> convex
--
--   An example with a different truth object or different loss preferences declares its
--   own @instance InferenceSignature ...@ in its F layer (as MNIST does for its @LogVec Bool@
--   satisfaction, reading it out with @logVecPTrue@ and using @lossKnow = negLog@).
module F_Inferential.InferenceInterpretation () where

import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Library.Convex (convex)
import F_Inferential.Library.CrossEntropy (crossEntropy)
import F_Inferential.Library.Softplus (softplus)
import qualified Torch

instance InferenceSignature Torch.Tensor where
  type Loss Torch.Tensor = Torch.Tensor
  lossKnow = softplus
  lossData = crossEntropy
  lossComb = convex
