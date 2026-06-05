{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the MNIST multi-digit example: honest labeled
--   metrics — sum-accuracy (train/test), per-digit accuracy (the four latent digits, scored
--   against true labels), and the mean satisfaction probability @P(sum=n)@ (read in @LogVec@ via
--   'logVecPTrue'). Predictions are the argmax of the @LogVec@ logits (no softmax); a two-digit
--   number is @10*argmax(hi) + argmax(lo)@.
module MnistMultiDigit.G_Statistical.Interpretation (report, multiReport) where

import A_Categorical.Monads.LogVecExpect (logVecLeafTensor)
import A_Categorical.Monads.Bridge (encode)
import A_Categorical.Monads.LogVec (LogVec)
import B_Logical.Interpretations.TensorBool (logVecPTrue)
import MnistMultiDigit.C_Domain.Interpretation ()
import MnistMultiDigit.C_Domain.Signature (MnistKlRel (..))
import MnistMultiDigit.D_Grammatical.Signature (multiFormula)
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
          ("Digit-acc", digitAcc),
          ("Mean P(sum=n)", meanConfidence theta ds 100)
        ]

-- | Mean @P(n = number(x1,x2)+number(y1,y2))@ over a capped test subset, in ONE batched @LogVec@
--   pass: forward the first @cap@ quadruples together and read 'logVecPTrue' (the marginalization
--   is the log-space convolution of the four digit leaves; no joint), then average.
meanConfidence :: Weights -> MultiDataset -> Int -> Double
meanConfidence theta ds cap =
  let n = min cap (length (testSums ds))
   in if n == 0
        then 0
        else
          let sliceN t = Torch.sliceDim 0 0 n 1 t
              oneHot = Torch.asTensor [[if s == k then 1.0 else 0.0 :: Float | k <- [0 .. 198]] | s <- take n (testSums ds)]
              obsCap = encode [0 .. 198] oneHot
              ps = logVecPTrue (multiFormula @LogVec theta (sliceN (testX1 ds), sliceN (testX2 ds), sliceN (testY1 ds), sliceN (testY2 ds), obsCap))
           in realToFrac (Torch.asValue (Torch.mean ps) :: Float)

-- | G-layer manifest piece for the Example.
report :: Weights -> MultiDataset -> IO Report
report theta ds = return (multiReport theta ds)
