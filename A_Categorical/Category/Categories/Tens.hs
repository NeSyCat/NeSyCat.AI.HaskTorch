-- | Objects of the TENS category (geometry paradigm).
module A_Categorical.Category.Categories.Tens
  ( TensObj,
  )
where

import qualified Torch

-- | Type membership in the TENS type system.
class TensObj a

instance TensObj Torch.Tensor

instance (TensObj a, TensObj b) => TensObj (a, b)

instance TensObj ()
