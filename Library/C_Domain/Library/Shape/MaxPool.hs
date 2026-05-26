-- | Shape (parameter space ℝ⁰): 2×2 max-pool, stride 2 — halves each spatial dim.
module C_Domain.Library.Shape.MaxPool (maxPool) where

import Torch (Tensor)
import qualified Torch.Functional as F

maxPool :: Tensor -> Tensor
maxPool x = F.maxPool2d (2, 2) (2, 2) (0, 0) (1, 1) F.Floor x
