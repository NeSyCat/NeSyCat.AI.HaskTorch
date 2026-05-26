-- | Tensor multiplicative monoid operation: otimes = pointwise multiplication.
--
-- In the logit encoding, the multiplicative monoid is ordinary real multiplication.
-- The neutral element is v1 = 1.0.
-- This is distinct from wedge (smooth min): otimes is bilinear, wedge is smooth-min.
module B_Logical.Connector.Otimes.Tensor (otimes) where

import qualified Torch

-- | Pointwise multiplication.  otimes = Torch.mul.
otimes :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
otimes = Torch.mul
