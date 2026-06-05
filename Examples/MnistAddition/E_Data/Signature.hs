-- | Data layer (E) — SIGNATURE for the MNIST example: the fixed FORMAT of the
--   data. @trainBatch@ is the differentiable training input (images x, images y, and the observed
--   sum already as @eta n@ -- a @LogVec Int@ leaf over [0..18], the distributional format the axiom
--   binds); the @*Img@/@*Lab@/@*Sums@ fields back the (argmax) accuracy report. What the data IS,
--   independent of how it is loaded (loading is the Loader's job, "MnistAddition.E_Data.Loader").
module MnistAddition.E_Data.Signature
  ( MnistDataset (..),
    Dataset,
  )
where

import A_Categorical.Monads.LogVec (LogVec)
import qualified Torch

-- | A dataset of MNIST image pairs and their observed sums (the sum as a @LogVec Int@ leaf).
data MnistDataset = MnistDataset
  { trainBatch :: (Torch.Tensor, Torch.Tensor, LogVec Int),
    trainXImg :: Torch.Tensor,
    trainYImg :: Torch.Tensor,
    trainSums :: [Int],
    testXImg :: Torch.Tensor,
    testYImg :: Torch.Tensor,
    testSums :: [Int],
    testXLab :: [Int],
    testYLab :: [Int]
  }

-- | The canonical dataset type the Example manifest refers to.
type Dataset = MnistDataset
