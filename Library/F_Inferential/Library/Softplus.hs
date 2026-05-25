-- | Softplus penalty:  pen(sat) = -log(sigma(sat)).
--   A smooth knowledge-loss morphism in Diff.
module F_Inferential.Library.Softplus (softplus) where

import qualified Torch

softplus :: Torch.Tensor -> Torch.Tensor
softplus sat = negate (logSigmoid sat)
  where
    logSigmoid x = negate (Torch.log (Torch.exp (negate x) `Torch.add` Torch.onesLike x))
