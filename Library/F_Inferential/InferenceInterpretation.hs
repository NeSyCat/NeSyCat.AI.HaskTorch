{-# LANGUAGE TypeFamilies #-}

-- | TensReal inference interpretation: assigns each role symbol of
--   'InferenceSignature' to a concrete loss morphism from the library.
--
--     lossKnow |-> softplus
--     lossData |-> crossEntropy
--     lossComb |-> convex
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
