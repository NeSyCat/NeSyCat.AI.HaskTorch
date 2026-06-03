-- | Data layer (E) — SIGNATURE for the Binary example: the fixed FORMAT of the
--   data (the 'BinaryDataset' record + the canonical 'Dataset' alias the manifest
--   refers to). What the data IS, independent of how it is produced — producing it
--   is the Loader's job ("Binary.E_Data.Loader").
module Binary.E_Data.Signature
  ( BinaryDataset (..),
    Dataset,
  )
where

import qualified Torch

-- | A binary classification dataset (circle-in-square). 50 train / 50 test. Only the POINTS
--   are stored: the label is a deterministic concept of the point (the circle test, computed
--   by @labelA@), so unlike MNIST's ground-truth digit labels it needs no separate field.
data BinaryDataset = BinaryDataset
  { trainData :: Torch.Tensor,
    testData :: Torch.Tensor
  }

-- | The canonical dataset type the Example manifest refers to.
type Dataset = BinaryDataset
