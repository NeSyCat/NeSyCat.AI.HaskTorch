{-# LANGUAGE TypeFamilies #-}

-- | Inference interpretations: assign each role symbol of 'InferenceSignature' to a
--   concrete loss morphism from the library. Two interpretations, one per truth object:
--
--     * @Torch.Tensor@ (logit satisfaction, the TensReal logic):
--         lossKnow |-> softplus   lossData |-> crossEntropy   lossComb |-> convex
--     * @OmegaP@ ([0,1] satisfaction, the fuzzy/LTN logic of "TensorProb"):
--         lossKnow |-> oneMinus   (1 - SAT)   -- the categorical analogue of softplus
--         lossData |-> crossEntropy            lossComb |-> convex
--
--   The @OmegaP@ pairing (probability-truth <-> @1 - SAT@) is general -- any example
--   whose axiom reads in [0,1] uses it -- so it lives here, not in any one example.
module F_Inferential.InferenceInterpretation () where

import B_Logical.Interpretations.TensorProb (OmegaP (..))
import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Library.Convex (convex)
import F_Inferential.Library.CrossEntropy (crossEntropy)
import F_Inferential.Library.NegLog (negLog)
import F_Inferential.Library.Softplus (softplus)
import qualified Torch

instance InferenceSignature Torch.Tensor where
  type Loss Torch.Tensor = Torch.Tensor
  lossKnow = softplus
  lossData = crossEntropy
  lossComb = convex

-- | The fuzzy/LTN reading: a satisfaction degree in [0,1], penalized by @-log SAT@.
--   (@negLog@, not @1 - SAT@: the log gives a steep gradient as SAT -> 0, which the
--   soft-marginalized MNIST axiom needs to avoid a collapsed local optimum; @1 - SAT@'s
--   bounded gradient is too weak there.)
instance InferenceSignature OmegaP where
  type Loss OmegaP = Torch.Tensor
  lossKnow (OmegaP sat) = negLog sat
  lossData (OmegaP p) (OmegaP y) = crossEntropy p y
  lossComb dataLoss knowLoss (OmegaP lam) = convex dataLoss knowLoss lam
