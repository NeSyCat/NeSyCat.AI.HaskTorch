-- | Shape (parameter space ℝ⁰): flatten all non-batch dims, @[B, …] -> [B, prod …]@.
module C_Domain.Library.Shape.Flatten (flatten) where

import Torch (Tensor)
import qualified Torch

flatten :: Tensor -> Tensor
flatten x = Torch.reshape [head (Torch.shape x), product (tail (Torch.shape x))] x
