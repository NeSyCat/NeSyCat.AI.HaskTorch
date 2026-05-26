-- | Smooth maximum (LogSumExp) over R-valued logits.
--
-- vee beta a b = log(exp(a*beta) + exp(b*beta)) / beta
--
-- As beta -> inf  this converges to  max(a, b).
-- As beta -> 0    this converges to  (a + b) / 2  (average).
-- Standard choice: beta = 1  (temperature-scaled softmax).
module B_Logical.Connector.Vee.Tensor (vee) where

import qualified Torch
import qualified Torch.Functional.Internal as F

-- | Smooth disjunction.  ParamsLogic Omega = Torch.Tensor (beta scalar).
vee :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor -> Torch.Tensor
vee betaT a b =
    let pa = a `Torch.mul` betaT
        pb = b `Torch.mul` betaT
     in F.logaddexp pa pb `Torch.div` betaT
