{-# LANGUAGE TypeFamilies #-}

-- | TensReal inference interpretation: assigns each role symbol of
--   'InferenceSignature' to a concrete loss morphism from the library.
--
--     lossKnow |-> softplus
--     lossData |-> crossEntropy
--     lossComb |-> convex
module E_Inferential.InferenceInterpretation () where

import E_Inferential.InferenceSignature (InferenceSignature (..))
import E_Inferential.Library.Convex (convex)
import E_Inferential.Library.CrossEntropy (crossEntropy)
import E_Inferential.Library.Softplus (softplus)
import qualified Torch

instance InferenceSignature Torch.Tensor where
  type Loss Torch.Tensor = Torch.Tensor
  lossKnow = softplus
  lossData = crossEntropy
  lossComb = convex
