-- | The @pMeanError@ aggregator (the LTN @\forall@ semantics): the generalized mean
--   of the /errors/ @(1 - x)@, mirrored back:
--
--     pMeanError p d xs = 1 - ( mean_d (1 - xs)^p )^(1/p)
--
--   over truth degrees in [0,1]. As @p@ grows it focuses the gradient on the
--   least-satisfied elements (the worst @1 - x@), which is why it is preferred over a
--   plain mean for a universal quantifier. The errors are clamped away from 0 (LTN
--   @stable@) so the @log@ stays finite; built only from ops already used in the codebase.
module B_Logical.Library.PMeanError (pMeanError) where

import B_Logical.Library.Stable (clampNotZero)
import qualified Torch

-- | @pMeanError p d xs@: @1 - pMean p d (1 - xs)@, the universal aggregator.
pMeanError :: Float -> Int -> Torch.Tensor -> Torch.Tensor
pMeanError p d xs =
  let n = Torch.shape xs !! d
      err = clampNotZero 1e-4 (Torch.onesLike xs `Torch.sub` xs)
      ep = powf p err
      m = Torch.sumDim (Torch.Dim d) Torch.RemoveDim Torch.Float ep `Torch.div` Torch.asTensor (fromIntegral n :: Float)
      r = powf (1 / p) m
   in Torch.onesLike r `Torch.sub` r
  where
    powf q t = Torch.exp (Torch.log t `Torch.mul` Torch.asTensor (q :: Float))
