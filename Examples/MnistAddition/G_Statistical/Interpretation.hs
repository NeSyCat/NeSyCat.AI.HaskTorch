{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the MNIST example: honest labeled
--   metrics — sum-accuracy (train/test) and digit accuracy (the latent digits, scored
--   against true labels). Predictions are the argmax of the @LogVec@ logits (no softmax).
module MnistAddition.G_Statistical.Interpretation (report, mnistReport) where

import A_Categorical.Monads.LogVecExpect (logVecLeafTensor)
import A_Categorical.Monads.LogVec (LogVec)
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.C_Domain.Signature (MnistKlFun (..))
import MnistAddition.E_Data.Signature (MnistDataset (..))
import G_Statistical.Report (Report (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

predDigits :: Weights -> Torch.Tensor -> [Int]
predDigits theta imgs =
  Torch.asValue (Torch.argmax (Torch.Dim 1) Torch.RemoveDim (logVecLeafTensor (digit @LogVec theta (pure imgs))))

fracEq :: [Int] -> [Int] -> Double
fracEq a b = fromIntegral (length (filter id (zipWith (==) a b))) / fromIntegral (max 1 (length a))

mnistReport :: Weights -> MnistDataset -> Report
mnistReport theta ds =
  Report
    [ ("Sum-acc(train)", fracEq (predSum (trainXImg ds) (trainYImg ds)) (trainSums ds)),
      ("Sum-acc(test)", fracEq (predSum (testXImg ds) (testYImg ds)) (testSums ds)),
      ("Digit-acc", fracEq (predDigits theta (testXImg ds) ++ predDigits theta (testYImg ds)) (testXLab ds ++ testYLab ds))
    ]
  where
    predSum xs ys = zipWith (+) (predDigits theta xs) (predDigits theta ys)

-- | G-layer manifest piece for the Example.
report :: Weights -> MnistDataset -> IO Report
report theta ds = return (mnistReport theta ds)
