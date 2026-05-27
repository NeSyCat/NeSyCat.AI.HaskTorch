-- | The generalized-mean aggregator @pMean@ (the LTN @\exists@ semantics):
--
--     pMean p d xs = ( mean_d xs^p )^(1/p)
--
--   over truth degrees in [0,1]. As @p -> \infty@ it approaches @max@; @p = 1@ is the
--   arithmetic mean. The power is taken as @exp (p * log x)@ (no dependence on
--   @Torch.pow@'s scalar-argument order) and the mean as @sumDim@ / size, so this uses
--   only ops already exercised in the codebase. Inputs are clamped away from 0
--   (LTN @stable@) so the @log@ stays finite.
module B_Logical.Library.PMean (pMean) where

import B_Logical.Library.Stable (clampNotZero)
import qualified Torch

-- | @pMean p d xs@: the generalized mean of order @p@ over dimension @d@.
pMean :: Float -> Int -> Torch.Tensor -> Torch.Tensor
pMean p d xs =
  let n = Torch.shape xs !! d
      xp = powf p (clampNotZero 1e-4 xs)
      m = Torch.sumDim (Torch.Dim d) Torch.RemoveDim Torch.Float xp `Torch.div` Torch.asTensor (fromIntegral n :: Float)
   in powf (1 / p) m
  where
    powf q t = Torch.exp (Torch.log t `Torch.mul` Torch.asTensor (q :: Float))
