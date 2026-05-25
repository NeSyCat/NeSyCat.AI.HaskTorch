{-# LANGUAGE TypeApplications #-}

-- | Inference layer (F) — INTERPRETATION for the MNIST example: the training
--   config and the objective, plus MNIST's interpretation of the (REUSED) inference
--   signature's knowledge loss. The signature stays the same as binary's; only the
--   interpretation differs -- here the penalty is the categorical
--   negative-log-likelihood over the 19 sum classes, where binary interprets it as
--   'softplus'. This is the single seam that turns the GeomU sum-logits into a
--   learning signal.
--
--   Crucially the softmax lives /here/, in the inference interpretation -- never in
--   GeomU. And @softmax@ over the 19 sum-logits is exactly the law of total
--   probability @P(sum=n) = sum_{d1+d2=n} softmax(lx)(d1) . softmax(ly)(d2)@ (the
--   partition factorizes), so this penalty is the principled one.
module MnistAddition.F_Inferential.Interpretation (objective, trainConfig, mnistKnowLoss) where

import MnistAddition.D_Grammatical.InterpretationTens (mnistSumLogits)
import MnistAddition.E_Data.Signature (MnistDataset (..))
import F_Inferential.Library.NegLog (negLog)
import C_Domain.Models.MnistCNN (ParamsCNN)
import qualified Torch
import qualified Torch.Functional as F

-- | (epochs, learning rate).
trainConfig :: (Int, Float)
trainConfig = (200, 0.001)

-- | The loss the trainer minimizes: the inference penalty (categorical NLL) of the
--   GeomU sum-logits term over the observed sums. Reaches the net only through the
--   grammatical term @digit(x)+digit(y)@ ('mnistSumLogits').
objective :: MnistDataset -> ParamsCNN -> Torch.Tensor
objective ds theta =
  let (xB, yB, oneHotObserved) = trainBatch ds
   in mnistKnowLoss (mnistSumLogits theta (xB, yB)) oneHotObserved

-- | @mnistKnowLoss sumLogits oneHotObserved@: categorical NLL of the observed sums
--   under the predicted sum-distribution. softmax (the push to probabilities) is
--   applied to the @[B,19]@ logits, the observed sum is selected by the one-hot, and
--   the @forall@ over pairs is the mean of @-log P@.
mnistKnowLoss :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
mnistKnowLoss sumLogits oneHotObserved =
  let probs = F.softmax (F.Dim 1) sumLogits -- [B,19]  (softmax: only here, in the penalty)
      pObserved =
        Torch.sumDim (Torch.Dim 1) Torch.RemoveDim Torch.Float (oneHotObserved `Torch.mul` probs) -- [B]
   in Torch.mean (negLog pObserved) -- forall over pairs = mean of -log P(sum=n)
