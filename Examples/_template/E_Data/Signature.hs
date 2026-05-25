-- | Data layer (E) — SIGNATURE for the Template example: the fixed FORMAT of the
--   data (the 'TemplateDataset' type + the canonical 'Dataset' alias the manifest
--   refers to). What the data IS, independent of how it is produced — producing it
--   is the Loader's job ("Template.E_Data.Loader").
module Template.E_Data.Signature
  ( TemplateDataset (..),
    Dataset,
  )
where

import qualified Torch

newtype TemplateDataset = TemplateDataset {inputs :: Torch.Tensor}

-- | The canonical dataset type the Example manifest refers to.
type Dataset = TemplateDataset
