-- | Data layer (E) — SIGNATURE for the MNIST multi-digit example: the fixed FORMAT of the data.
--   @trainBatch@ is the differentiable training input (the four image groups + the observed sum
--   already as @eta n@ -- a @LogVec Int@ leaf over [0..198]); the @*Img@/@*Lab@/@*Sums@ fields back
--   the (argmax) accuracy report. Two TWO-digit numbers per example: A = (x1 hi, x2 lo),
--   B = (y1 hi, y2 lo), observed sum N = number A + number B in 0..198.
module MnistMultiDigit.E_Data.Signature
  ( MultiDataset (..),
    Dataset,
  )
where

import A_Categorical.Monads.LogVec (LogVec)
import qualified Torch

-- | A dataset of MNIST image quadruples and their observed sums (the sum as a @LogVec Int@ leaf).
--   Each @*Img@ field is a @[N,1,28,28]@ batch; the four digit-label lists back per-digit accuracy.
data MultiDataset = MultiDataset
  { trainBatch :: (Torch.Tensor, Torch.Tensor, Torch.Tensor, Torch.Tensor, LogVec Int),
    trainX1 :: Torch.Tensor,
    trainX2 :: Torch.Tensor,
    trainY1 :: Torch.Tensor,
    trainY2 :: Torch.Tensor,
    trainSums :: [Int],
    testX1 :: Torch.Tensor,
    testX2 :: Torch.Tensor,
    testY1 :: Torch.Tensor,
    testY2 :: Torch.Tensor,
    testSums :: [Int],
    testLabs :: ([Int], [Int], [Int], [Int]) -- the four digit-label lists (x1,x2,y1,y2)
  }

-- | The canonical dataset type the Example manifest refers to.
type Dataset = MultiDataset
