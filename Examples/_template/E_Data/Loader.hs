-- | G (data) layer for the Template example: prepare the dataset (already in
--   tensor form). This stub returns a constant tensor; replace with your loader
--   (read files from @data/@, build batches, ...).
module Template.E_Data.Loader
  ( TemplateDataset (..),
    Dataset,
    loadTemplateDataset,
    loadData,
  )
where

import qualified Torch

newtype TemplateDataset = TemplateDataset {inputs :: Torch.Tensor}

loadTemplateDataset :: IO TemplateDataset
loadTemplateDataset = pure (TemplateDataset (Torch.ones' [8, 1]))

-- | E-layer manifest pieces for the Example (canonical names).
type Dataset = TemplateDataset

loadData :: IO Dataset
loadData = loadTemplateDataset
