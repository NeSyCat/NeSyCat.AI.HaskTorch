-- | Strict less-than predicate lifted to tensor truth (logit encoding).
module B_Logical.Comparator.Lt.Tensor ((.<)) where

import qualified Torch

infix 4 .<

-- | Tensor strict less-than: result as logit tensor.
(.<) :: (Ord a) => a -> a -> Torch.Tensor
x .< y
  | x < y     = Torch.asTensor [(1.0 / 0.0) :: Float]
  | otherwise = Torch.asTensor [(-1.0 / 0.0) :: Float]
