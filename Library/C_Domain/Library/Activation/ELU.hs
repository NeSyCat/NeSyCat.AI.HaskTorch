-- | Activation (parameter space ℝ⁰): the ELU nonlinearity with @α = 1@.
module C_Domain.Library.Activation.ELU (elu) where

import Torch (Tensor)
import qualified Torch.Functional as F

elu :: Tensor -> Tensor
elu = F.elu (1.0 :: Float)
