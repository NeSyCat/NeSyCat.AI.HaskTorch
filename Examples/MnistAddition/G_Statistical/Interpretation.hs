{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the MNIST example: honest labeled
--   metrics — sum-accuracy (train/test), digit accuracy (the latent digits, scored
--   against true labels), and the mean @Dist@ probability @P(sum=n)@. Predictions are
--   the argmax of the @LogVec@ logits (no softmax).
module MnistAddition.G_Statistical.Interpretation (report, mnistReport) where

import A_Categorical.Category.Monads.LogVecExpect (logVecLeafTensor)
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import A_Categorical.Category.Monads.LogVec (LogVec)
import A_Categorical.Category.Monads.Dist (Dist)
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.C_Domain.Signature (MnistKlRel (..))
import MnistAddition.D_Grammatical.Signature (mnistFormula)
import MnistAddition.E_Data.Signature (MnistDataset (..))
import G_Statistical.Report (Report (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

predDigits :: Weights -> Torch.Tensor -> [Int]
predDigits theta imgs =
  Torch.asValue (Torch.argmax (Torch.Dim 1) Torch.RemoveDim (logVecLeafTensor (digit @LogVec theta imgs)))

fracEq :: [Int] -> [Int] -> Double
fracEq a b = fromIntegral (length (filter id (zipWith (==) a b))) / fromIntegral (max 1 (length a))

mnistReport :: Weights -> MnistDataset -> Report
mnistReport theta ds =
  Report
    [ ("Sum-acc(train)", fracEq (predSum (trainXImg ds) (trainYImg ds)) (trainSums ds)),
      ("Sum-acc(test)", fracEq (predSum (testXImg ds) (testYImg ds)) (testSums ds)),
      ("Digit-acc", fracEq (predDigits theta (testXImg ds) ++ predDigits theta (testYImg ds)) (testXLab ds ++ testYLab ds)),
      ("Mean P(sum=n)", meanConfidence theta ds 100)
    ]
  where
    predSum xs ys = zipWith (+) (predDigits theta xs) (predDigits theta ys)

-- | Mean @P(add(x,y) = digit(x)+digit(y))@ over a capped test subset: the per-pair @Dist@
--   reading ('mnistFormula' at @\@Dist@, then @distPTrue@), averaged. (G does its own
--   per-pair reading, like binary's report, rather than the whole-dataset axiom.)
meanConfidence :: Weights -> MnistDataset -> Int -> Double
meanConfidence theta ds cap =
  let n = min cap (length (testSums ds))
      slice img i = Torch.sliceDim 0 i (i + 1) 1 img
      prob i = distPTrue (mnistFormula @Dist theta (slice (testXImg ds) i, slice (testYImg ds) i, pure (testSums ds !! i)))
      ps = [prob i | i <- [0 .. n - 1]]
   in if null ps then 0 else sum ps / fromIntegral (length ps)

-- | G-layer manifest piece for the Example.
report :: Weights -> MnistDataset -> IO Report
report theta ds = return (mnistReport theta ds)
