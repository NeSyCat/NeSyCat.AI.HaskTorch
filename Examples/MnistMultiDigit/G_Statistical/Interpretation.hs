{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the MNIST multi-digit example: honest labeled
--   metrics — sum-accuracy (train/test) and per-digit accuracy (the four latent digits, scored
--   against true labels). Predictions are the argmax of the @LogVec@ logits (no softmax); a
--   two-digit number is @10*argmax(hi) + argmax(lo)@.
module MnistMultiDigit.G_Statistical.Interpretation (report, multiReport) where

import A_Categorical.Monads.LogVecExpect (logVecLeafTensor)
import A_Categorical.Monads.LogVec (LogVec)
import MnistMultiDigit.C_Domain.Interpretation ()
import MnistMultiDigit.C_Domain.Signature (MnistKlFun (..))
import MnistMultiDigit.E_Data.Signature (MultiDataset (..))
import G_Statistical.Report (Report (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

predDigits :: Weights -> Torch.Tensor -> [Int]
predDigits theta imgs =
  Torch.asValue (Torch.argmax (Torch.Dim 1) Torch.RemoveDim (logVecLeafTensor (digit @LogVec theta imgs)))

fracEq :: [Int] -> [Int] -> Double
fracEq a b = fromIntegral (length (filter id (zipWith (==) a b))) / fromIntegral (max 1 (length a))

multiReport :: Weights -> MultiDataset -> Report
multiReport theta ds =
  let predNum hi lo = zipWith (\h l -> 10 * h + l) (predDigits theta hi) (predDigits theta lo)
      predSum x1 x2 y1 y2 = zipWith (+) (predNum x1 x2) (predNum y1 y2)
      (l1, l2, l3, l4) = testLabs ds
      digitAcc =
        fracEq
          (predDigits theta (testX1 ds) ++ predDigits theta (testX2 ds) ++ predDigits theta (testY1 ds) ++ predDigits theta (testY2 ds))
          (l1 ++ l2 ++ l3 ++ l4)
   in Report
        [ ("Sum-acc(train)", fracEq (predSum (trainX1 ds) (trainX2 ds) (trainY1 ds) (trainY2 ds)) (trainSums ds)),
          ("Sum-acc(test)", fracEq (predSum (testX1 ds) (testX2 ds) (testY1 ds) (testY2 ds)) (testSums ds)),
          ("Digit-acc", digitAcc)
        ]

-- | G-layer manifest piece for the Example.
report :: Weights -> MultiDataset -> IO Report
report theta ds = return (multiReport theta ds)
