{-# LANGUAGE TypeApplications #-}

-- | Data for the Binary classification example, prepared independently of any
--   training or loss. A dataset is just tensors in the fixed format (points and
--   their ground-truth labels); the inferential layer consumes it without knowing
--   how it was produced. Here the points are sampled and labelled by the domain's
--   own @labelA@ (the circle-in-square concept) -- data preparation, not training.
module Binary.E_Data.Loader
  ( BinaryDataset (..),
    Dataset,
    generateBinaryDataset,
    loadData,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import Binary.C_Domain.Interpretation ()
import Binary.C_Domain.Signature (BinaryRel (..))
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))

-- | A binary classification dataset (circle-in-square). 50 train / 50 test.
data BinaryDataset = BinaryDataset
  { trainData :: Torch.Tensor,
    trainLabels :: Torch.Tensor,
    testData :: Torch.Tensor,
    testLabels :: Torch.Tensor
  }

-- | Sample 100 random points in [0,1]^2; labels come from @labelA \@GeomU@
--   (batched), so the ground-truth concept lives only in the domain interpretation.
generateBinaryDataset :: IO BinaryDataset
generateBinaryDataset = do
  dataset <- Torch.toDevice (Device CPU 0) <$> Torch.randIO' [100, 2]
  let logits = labelA @GeomU dataset
      labels = Torch.toType Torch.Float (logits `Torch.gt` Torch.asTensor (0.0 :: Float))
  return
    BinaryDataset
      { trainData = Torch.sliceDim 0 0 50 1 dataset,
        trainLabels = Torch.reshape [50, 1] (Torch.sliceDim 0 0 50 1 labels),
        testData = Torch.sliceDim 0 0 50 1 (Torch.sliceDim 0 50 100 1 dataset),
        testLabels = Torch.reshape [50, 1] (Torch.sliceDim 0 0 50 1 (Torch.sliceDim 0 50 100 1 labels))
      }

-- | E-layer manifest pieces for the Example (canonical names).
type Dataset = BinaryDataset

loadData :: IO Dataset
loadData = generateBinaryDataset
