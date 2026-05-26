-- | Tensor top: top = +infinity (logit space).
--
-- In the logit encoding, +inf represents certainty of truth.
-- It is the neutral element of smooth min (neg-LogSumExp): wedge(+inf, x) = x.
module B_Logical.Connector.Top.Tensor (top) where

import qualified Torch

-- | Greatest element: +∞ as a rank-1 tensor of shape [1].
top :: Torch.Tensor
top = Torch.asTensor [(1.0 / 0.0) :: Float]
