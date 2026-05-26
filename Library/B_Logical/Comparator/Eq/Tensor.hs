-- | Equality predicate lifted to tensor truth (logit encoding).
--
-- Returns +inf (True) when a == b, -inf (False) otherwise.
-- Useful when the result must flow into smooth connectives.
module B_Logical.Comparator.Eq.Tensor ((.==)) where

import qualified Torch

infix 4 .==

-- | Tensor equality: scalar comparison, result as logit tensor.
(.==) :: (Eq a) => a -> a -> Torch.Tensor
x .== y
  | x == y    = Torch.asTensor [(1.0 / 0.0) :: Float]
  | otherwise = Torch.asTensor [(-1.0 / 0.0) :: Float]
