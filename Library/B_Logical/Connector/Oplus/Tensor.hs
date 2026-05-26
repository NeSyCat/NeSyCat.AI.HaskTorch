-- | Tensor additive monoid operation: oplus = pointwise addition.
--
-- In the logit encoding, the additive monoid is ordinary real addition.
-- The neutral element is v0 = 0.0.
-- This is distinct from vee (smooth max): oplus is linear, vee is smooth-max.
module B_Logical.Connector.Oplus.Tensor (oplus) where

import qualified Torch

-- | Pointwise addition.  oplus = Torch.add.
oplus :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
oplus = Torch.add
