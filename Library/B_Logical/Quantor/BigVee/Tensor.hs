-- | Existential quantifier for tensor truth (GeomU universe): smooth maximum.
--
-- bigVee beta guard phi = (LogSumExp(phi(guard)*beta, dim=0) - log N) / beta
--
-- As beta -> inf  this converges to  max over the batch.
-- PyTorch broadcasts phi over the batch dimension automatically.
-- This is the geometric analogue of classical exists.
module B_Logical.Quantor.BigVee.Tensor (bigVee) where

import Data.Functor.Identity (Identity (..))
import qualified Torch
import qualified Torch.Functional.Internal as F

-- | Smooth existential quantifier.  ParamsLogic Omega = Torch.Tensor (beta scalar).
bigVee
    :: Torch.Tensor               -- ^ beta (smoothing parameter)
    -> Torch.Tensor               -- ^ guard: batch tensor of shape [N, ...]
    -> (Torch.Tensor -> Identity Torch.Tensor)  -- ^ phi: predicate
    -> Identity Torch.Tensor
bigVee betaT guard phi =
    let result  = runIdentity (phi guard)
        n       = head (Torch.shape guard)
        lse     = F.logsumexp (result `Torch.mul` betaT) 0 False
        reduced = (lse `Torch.sub` Torch.log (Torch.asTensor (fromIntegral n :: Float))) `Torch.div` betaT
     in Identity (Torch.reshape [1] reduced)
  where runIdentity (Identity x) = x
