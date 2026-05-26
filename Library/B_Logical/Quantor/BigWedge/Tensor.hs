-- | Universal quantifier for tensor truth (GeomU universe): smooth minimum.
--
-- bigWedge beta guard phi = neg(LogSumExp(-phi(guard)*beta, dim=0) - log N) / beta
--
-- As beta -> inf  this converges to  min over the batch.
-- PyTorch broadcasts phi over the batch dimension automatically.
-- This is the geometric analogue of classical forall.
module B_Logical.Quantor.BigWedge.Tensor (bigWedge) where

import Data.Functor.Identity (Identity (..))
import qualified Torch
import qualified Torch.Functional.Internal as F

-- | Smooth universal quantifier.  ParamsLogic Omega = Torch.Tensor (beta scalar).
bigWedge
    :: Torch.Tensor               -- ^ beta (smoothing parameter)
    -> Torch.Tensor               -- ^ guard: batch tensor of shape [N, ...]
    -> (Torch.Tensor -> Identity Torch.Tensor)  -- ^ phi: predicate
    -> Identity Torch.Tensor
bigWedge betaT guard phi =
    let result   = runIdentity (phi guard)
        n        = head (Torch.shape guard)
        negResult = negate result
        lse      = F.logsumexp (negResult `Torch.mul` betaT) 0 False
        reduced  = negate ((lse `Torch.sub` Torch.log (Torch.asTensor (fromIntegral n :: Float))) `Torch.div` betaT)
     in Identity (Torch.reshape [1] reduced)
  where runIdentity (Identity x) = x
