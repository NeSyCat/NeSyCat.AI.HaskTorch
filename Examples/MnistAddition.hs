{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The MNIST single-digit-addition experiment as a single 'Example' instance:
--   it wires the MNIST interpretations (gamma: @digit@/@plus@/@eqNat@), its
--   objective (epsilon: 'lossForMnist' = softplus of the GeomU axiom) and its
--   metrics (zeta: sum-/digit-accuracy via argmax of the logits, plus the MeasU
--   @P(sum=n)@ confidence). 'runExample' does the rest -- same button as binary.
module Examples.MnistAddition (MnistAdd) where

import A_Categorical.CategoricalInterpretation (GeomU)
import C_Domain.Examples.MnistAddition.Interpretation ()
import C_Domain.Examples.MnistAddition.Signature (MnistKlRel (..))
import C_Domain.Models.MnistCNN (ParamsCNN, ParamsCNNSpec (..))
import D_Grammatical.Examples.MnistAddition.IntpData (mnistProb)
import Data.Functor.Identity (runIdentity)
import E_Inferential.Examples.MnistAddition.Train (MnistDataset (..), loadMnistDataset, lossForMnist)
import Example (Example (..))
import F_Statistical.Report (BenchmarkReport (..))
import qualified Torch

-- | Tag type selecting the MNIST-addition experiment.
data MnistAdd

instance Example MnistAdd where
  type Params MnistAdd = ParamsCNN
  type Spec MnistAdd = ParamsCNNSpec
  type Data MnistAdd = MnistDataset

  name = "MNIST single-digit addition (axiom-only; digits learned from sums)"
  spec = ParamsCNNSpec
  trainConfig = (30, 0.001)
  loadData = loadMnistDataset
  objective ds theta = lossForMnist ds theta
  report theta ds = return (mnistReport theta ds)

-- | Predicted digits = argmax of the GeomU logits (no softmax; logits give the
--   argmax, exactly as for the deterministic reading).
predDigits :: ParamsCNN -> Torch.Tensor -> [Int]
predDigits theta imgs =
  Torch.asValue (Torch.argmax (Torch.Dim 1) Torch.RemoveDim (runIdentity (digit @GeomU theta imgs)))

fracEq :: [Int] -> [Int] -> Double
fracEq a b = fromIntegral (length (filter id (zipWith (==) a b))) / fromIntegral (max 1 (length a))

-- | sum-accuracy on Acc fields, digit-accuracy on the F1/precision fields, and
--   the mean MeasU @P(sum=n)@ as the (positive) confidence.
mnistReport :: ParamsCNN -> MnistDataset -> BenchmarkReport
mnistReport theta ds =
  let predSum xs ys = zipWith (+) (predDigits theta xs) (predDigits theta ys)
      trainSumAcc = fracEq (predSum (trainXImg ds) (trainYImg ds)) (trainSums ds)
      testSumAcc = fracEq (predSum (testXImg ds) (testYImg ds)) (testSums ds)
      testDigAcc =
        fracEq
          (predDigits theta (testXImg ds) ++ predDigits theta (testYImg ds))
          (testXLab ds ++ testYLab ds)
   in BenchmarkReport
        { reportAccTrain = trainSumAcc,
          reportAccTest = testSumAcc,
          reportF1 = testDigAcc, -- digit accuracy (latent digits, scored against true labels)
          reportPrecision = testDigAcc,
          reportConfPos = meanConfidence theta ds 100,
          reportConfNeg = 0
        }

-- | Mean @P(add(x,y) = digit(x)+digit(y))@ over a capped test subset: the MeasU
--   ('Dist') probability reading of the per-pair axiom.
meanConfidence :: ParamsCNN -> MnistDataset -> Int -> Double
meanConfidence theta ds cap =
  let n = min cap (length (testSums ds))
      slice img i = Torch.sliceDim 0 i (i + 1) 1 img -- [1,1,28,28]
      ps =
        [ mnistProb theta (slice (testXImg ds) i, slice (testYImg ds) i, testSums ds !! i)
          | i <- [0 .. n - 1]
        ]
   in if null ps then 0 else sum ps / fromIntegral (length ps)
