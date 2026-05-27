{-# LANGUAGE TypeFamilies #-}

-- | Inference layer (F) — INTERPRETATION for the MNIST example: ONLY the choice of
--   losses (the 'InferenceSignature' instance for MNIST's satisfaction object) plus the
--   training hyper-parameters. It is BLIND to the model, the data and the axiom -- the
--   generic objective @lossKnow . sat@ (in "Example") combines the @sat@ exported by D
--   with the @lossKnow@ chosen here. No objective, no batching, no forward pass.
module MnistAddition.F_Inferential.Interpretation (trainConfig) where

import B_Logical.Interpretations.TensorProb (OmegaP (..))
import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Library.Convex (convex)
import F_Inferential.Library.CrossEntropy (crossEntropy)
import F_Inferential.Library.NegLog (negLog)
import qualified Torch

-- | MNIST's loss choices on the [0,1] satisfaction @OmegaP@: knowledge loss = @negLog@
--   (so @J = -log SAT@), data loss = cross-entropy, combination = convex. MNIST is pure
--   knowledge (no labels), so only @lossKnow@ is exercised (lambda = 1).
instance InferenceSignature OmegaP where
  type Loss OmegaP = Torch.Tensor
  lossKnow (OmegaP s) = negLog s
  lossData (OmegaP p) (OmegaP y) = crossEntropy p y
  lossComb dataLoss knowLoss (OmegaP lam) = convex dataLoss knowLoss lam

-- | (epochs, learning rate). LTN's small-data single-digit setup: 20 epochs.
trainConfig :: (Int, Float)
trainConfig = (20, 0.001)
