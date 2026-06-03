{-# LANGUAGE TypeApplications #-}

-- | Data layer (E) — the LOADER for the Binary example: produces a 'BinaryDataset'
--   (the format from "Binary.E_Data.Signature"), prepared independently of any
--   training or loss. Points are sampled and labelled by the domain's own
--   @labelA \@GeomU@ (the circle-in-square concept) -- data preparation, not training.
module Binary.E_Data.Loader
  ( generateBinaryDataset,
    loadData,
    batches,
  )
where

import Binary.E_Data.Signature (BinaryDataset (..), Dataset)
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))

-- | Sample 100 random points in [0,1]^2 and split 50/50. Only the points are stored; their
--   labels are the circle-in-square concept, computed on demand by @labelA@ (in the formula)
--   and by the report (G) -- so the loader does no labelling.
generateBinaryDataset :: IO BinaryDataset
generateBinaryDataset = do
  dataset <- Torch.toDevice (Device CPU 0) <$> Torch.randIO' [100, 2]
  return
    BinaryDataset
      { trainData = Torch.sliceDim 0 0 50 1 dataset,
        testData = Torch.sliceDim 0 50 100 1 dataset
      }

-- | E-layer manifest piece for the Example: how to obtain the data.
loadData :: IO Dataset
loadData = generateBinaryDataset

-- | Full batch: Binary trains on all 50 training points at once, so one batch = the
--   training tensor (the @Batch@ the axiom consumes; the epoch is unused).
batches :: Int -> BinaryDataset -> [Torch.Tensor]
batches _ ds = [trainData ds]
