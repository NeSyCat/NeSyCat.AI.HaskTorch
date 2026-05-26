-- | Multiplicative quantifier for tensor truth (GeomU universe).
--
-- In the logit encoding, the multiplicative monoid is (R, *, 1).
-- bigOtimes guard phi = product of phi(x) over x in guard.
-- This corresponds to a log-linear (product-of-experts) model:
-- log P = sum of logits = sum of log potentials.
--
-- Currently not formally supported in the typeclass instance; this module
-- provides the raw reduction function for experimentation.
module B_Logical.Quantor.BigOtimes.Tensor (bigOtimes) where

import Data.Functor.Identity (Identity (..))
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | Multiplicative reduction: product of phi(guard) over the batch dimension.
bigOtimes
    :: Torch.Tensor                              -- ^ guard: batch tensor [N, ...]
    -> (Torch.Tensor -> Identity Torch.Tensor)   -- ^ phi: predicate
    -> Identity Torch.Tensor
bigOtimes guard phi =
    let result = runIdentity (phi guard)
     in Identity (Torch.reshape [1] (FI.prodAll result Torch.Float))
  where runIdentity (Identity x) = x
