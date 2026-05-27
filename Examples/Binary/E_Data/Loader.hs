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

import A_Categorical.CategoricalInterpretation (GeomU)
import Binary.C_Domain.Interpretation ()
import Binary.C_Domain.Signature (BinaryRel (..))
import Binary.E_Data.Signature (BinaryDataset (..), Dataset)
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))

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

-- | E-layer manifest piece for the Example: how to obtain the data.
loadData :: IO Dataset
loadData = generateBinaryDataset

-- | Full batch: Binary trains on all 50 training points at once, so one batch = the
--   training tensor (the @Batch@ the axiom consumes; the epoch is unused).
batches :: Int -> BinaryDataset -> [Torch.Tensor]
batches _ ds = [trainData ds]
