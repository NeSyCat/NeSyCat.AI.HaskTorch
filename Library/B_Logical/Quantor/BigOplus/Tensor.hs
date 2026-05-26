-- | Additive quantifier for tensor truth (GeomU universe).
--
-- In the logit encoding, the additive monoid is (R, +, 0).
-- bigOplus guard phi = sum of phi(x) over x in guard (i.e. mean/sum reduction).
-- Note: this is NOT the same as bigVee (smooth max).  It accumulates evidence
-- by linear summation rather than soft-max selection.
--
-- Currently not formally supported in the typeclass instance; this module
-- provides the raw reduction function for experimentation.
module B_Logical.Quantor.BigOplus.Tensor (bigOplus) where

import Data.Functor.Identity (Identity (..))
import qualified Torch

-- | Additive reduction: sum phi(guard) over the batch dimension.
bigOplus
    :: Torch.Tensor                              -- ^ guard: batch tensor [N, ...]
    -> (Torch.Tensor -> Identity Torch.Tensor)   -- ^ phi: predicate
    -> Identity Torch.Tensor
bigOplus guard phi =
    let result = runIdentity (phi guard)
     in Identity (Torch.reshape [1] (Torch.sumAll result))
  where runIdentity (Identity x) = x
