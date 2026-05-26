-- | Smooth minimum (De Morgan dual of LogSumExp) over R-valued logits.
--
-- wedge beta a b = neg (vee beta (neg a) (neg b))
--               = -(log(exp(-a*beta) + exp(-b*beta)) / beta)
--
-- As beta -> inf  this converges to  min(a, b).
-- This is the default implementation from LogicalSignature via De Morgan.
module B_Logical.Connector.Wedge.Tensor (wedge) where

import qualified Torch
import qualified Torch.Functional.Internal as F

-- | Smooth conjunction.  ParamsLogic Omega = Torch.Tensor (beta scalar).
wedge :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor -> Torch.Tensor
wedge betaT a b =
    let na = negate a
        nb = negate b
        pa = na `Torch.mul` betaT
        pb = nb `Torch.mul` betaT
     in negate (F.logaddexp pa pb `Torch.div` betaT)
