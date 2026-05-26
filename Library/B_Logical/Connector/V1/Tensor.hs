-- | Tensor multiplicative unit: v1 = 1.0
--
-- Neutral element of otimes = Torch.mul: 1 * x = x.
module B_Logical.Connector.V1.Tensor (v1) where

import qualified Torch

-- | Multiplicative identity: 1.0 as a rank-1 tensor of shape [1].
v1 :: Torch.Tensor
v1 = Torch.asTensor [1.0 :: Float]
