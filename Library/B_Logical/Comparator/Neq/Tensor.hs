-- | Inequality predicate lifted to tensor truth (logit encoding).
module B_Logical.Comparator.Neq.Tensor ((./=)) where

import qualified Torch

infix 4 ./=

-- | Tensor inequality: scalar comparison, result as logit tensor.
(./=) :: (Eq a) => a -> a -> Torch.Tensor
x ./= y
  | x /= y    = Torch.asTensor [(1.0 / 0.0) :: Float]
  | otherwise = Torch.asTensor [(-1.0 / 0.0) :: Float]
