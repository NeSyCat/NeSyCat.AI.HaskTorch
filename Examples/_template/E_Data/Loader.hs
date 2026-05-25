-- | Data layer (E) — the LOADER for the Template example: produces a
--   'TemplateDataset' (the format from "Template.E_Data.Signature"). This stub
--   returns a constant tensor; replace with your loader (read files from @data/@,
--   build batches, ...).
module Template.E_Data.Loader
  ( loadTemplateDataset,
    loadData,
  )
where

import Template.E_Data.Signature (Dataset, TemplateDataset (..))
import qualified Torch

loadTemplateDataset :: IO TemplateDataset
loadTemplateDataset = pure (TemplateDataset (Torch.ones' [8, 1]))

-- | E-layer manifest piece for the Example: how to obtain the data.
loadData :: IO Dataset
loadData = loadTemplateDataset
