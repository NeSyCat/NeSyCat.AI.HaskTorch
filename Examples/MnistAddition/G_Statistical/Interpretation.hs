{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the MNIST example: honest labeled
--   metrics — sum-accuracy (train/test), digit accuracy (the latent digits, scored
--   against true labels), and the mean satisfaction probability @P(sum=n)@ (read in
--   @LogVec@ via 'logVecPTrue'). Predictions are the argmax of the @LogVec@ logits (no softmax).
module MnistAddition.G_Statistical.Interpretation (report, mnistReport) where

import A_Categorical.Monads.LogVecExpect (logVecLeafTensor)
import A_Categorical.Monads.Bridge (encode)
import A_Categorical.Monads.LogVec (LogVec)
import B_Logical.Interpretations.TensorBool (logVecPTrue)
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.C_Domain.Signature (MnistKlFun (..))
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

-- | Mean @P(add(x,y) = digit(x)+digit(y))@ over a capped test subset, in ONE batched @LogVec@
--   pass: forward the first @cap@ pairs together and read 'logVecPTrue' (the @LogVec@ twin of
--   @distPTrue@: @exp(logNum - logDen)@ over the leaf logits), then average. Numerically the same
--   per-pair probability as the @Dist@ reading (@digit \@Dist = decode . digit \@LogVec@), but one
--   batched forward instead of @cap@ single-image forwards.
meanConfidence :: Weights -> MnistDataset -> Int -> Double
meanConfidence theta ds cap =
  let n = min cap (length (testSums ds))
   in if n == 0
        then 0
        else
          let xCap = Torch.sliceDim 0 0 n 1 (testXImg ds)
              yCap = Torch.sliceDim 0 0 n 1 (testYImg ds)
              oneHot = Torch.asTensor [[if s == k then 1.0 else 0.0 :: Float | k <- [0 .. 18]] | s <- take n (testSums ds)]
              obsCap = encode [0 .. 18] oneHot
              ps = logVecPTrue (mnistFormula @LogVec theta (xCap, yCap, obsCap))
           in realToFrac (Torch.asValue (Torch.mean ps) :: Float)

-- | G-layer manifest piece for the Example.
report :: Weights -> MnistDataset -> IO Report
report theta ds = return (mnistReport theta ds)
