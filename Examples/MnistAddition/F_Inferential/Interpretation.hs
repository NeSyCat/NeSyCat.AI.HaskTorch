{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

-- | Inference layer (F) — INTERPRETATION for the MNIST example: ONLY the choice of
--   losses (the 'InferenceSignature' instance for MNIST's satisfaction object) plus the
--   training hyper-parameters. It is BLIND to the model, the data and the axiom -- the
--   generic objective @lossKnow . sat@ (in "Example") combines the @sat@ exported by D
--   with the @lossKnow@ chosen here. No objective, no batching, no forward pass.
module MnistAddition.F_Inferential.Interpretation (trainConfig) where

import A_Categorical.Category.Monads.LogVec (LogVec)
import B_Logical.Interpretations.TensorBool (logVecPTrue)
import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Library.Convex (convex)
import F_Inferential.Library.CrossEntropy (crossEntropy)
import F_Inferential.Library.NegLog (negLog)
import qualified Torch

-- | MNIST's loss choices on the satisfaction object @LogVec Bool@ (the GeomU sibling of
--   @Dist Bool@): read it out to a [0,1] degree with 'logVecPTrue' (the twin of @distPTrue@),
--   then knowledge loss = @negLog@ (so @J = -log SAT@). @-log(geomean s) = mean(-log s)@ is
--   the NLL. data loss = cross-entropy, combination = convex. MNIST is pure knowledge (no
--   labels), so only @lossKnow@ is exercised (lambda = 1).
instance InferenceSignature (LogVec Bool) where
  type Loss (LogVec Bool) = Torch.Tensor
  lossKnow m = negLog (logVecPTrue m)
  lossData m y = crossEntropy (logVecPTrue m) (logVecPTrue y)
  lossComb dataLoss knowLoss m = convex dataLoss knowLoss (logVecPTrue m)

-- | (epochs, learning rate). LTN's small-data single-digit setup: 20 epochs.
trainConfig :: (Int, Float)
trainConfig = (20, 0.001)
