-- | Smooth implication in logit space via De Morgan residual.
--
-- implies beta a b = vee beta (neg a) b
--                 = log(exp(-a*beta) + exp(b*beta)) / beta
--
-- This is the Lukasiewicz-style residual in the logit encoding.
module B_Logical.Connector.Implies.Tensor (implies) where

import qualified Torch
import qualified Torch.Functional.Internal as F

-- | Smooth implication.  ParamsLogic Omega = Torch.Tensor (beta scalar).
implies :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor -> Torch.Tensor
implies betaT a b =
    let na = negate a
        pa = na `Torch.mul` betaT
        pb = b  `Torch.mul` betaT
     in F.logaddexp pa pb `Torch.div` betaT
