-- | G (data) layer for the Template example: prepare the dataset (already in
--   tensor form). This stub returns a constant tensor; replace with your loader
--   (read files from @data/@, build batches, ...).
module Examples.Template.G_Data.Loader
  ( TemplateDataset (..),
    loadTemplateDataset,
  )
where

import qualified Torch

newtype TemplateDataset = TemplateDataset {inputs :: Torch.Tensor}

loadTemplateDataset :: IO TemplateDataset
loadTemplateDataset = pure (TemplateDataset (Torch.ones' [8, 1]))
