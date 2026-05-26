-- | Tensor additive unit: v0 = 0.0
--
-- Neutral element of oplus = Torch.add: 0 + x = x.
module B_Logical.Connector.V0.Tensor (v0) where

import qualified Torch

-- | Additive identity: 0.0 as a rank-1 tensor of shape [1].
v0 :: Torch.Tensor
v0 = Torch.asTensor [0.0 :: Float]
