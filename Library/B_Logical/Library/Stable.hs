-- | Numerically stable clamps for fuzzy aggregators (LTN's @stable=True@ trick).
--   Keep truth degrees away from the boundary so @log@ inside a p-mean stays finite:
--
--     clampNotZero eps x = (1 - eps) x + eps   -- push x in [0,1] up to >= eps
--     clampNotOne  eps x = (1 - eps) x          -- pull x in [0,1] down to <= 1 - eps
--
--   LTN uses @eps = 1e-4@. Built only from @mul@/@add@ against a broadcast scalar
--   tensor, so no scalar-argument-order assumptions about @*Scalar@.
module B_Logical.Library.Stable (clampNotZero, clampNotOne) where

import qualified Torch

-- | @(1 - eps) x + eps@: push degrees away from 0.
clampNotZero :: Float -> Torch.Tensor -> Torch.Tensor
clampNotZero eps x =
  (x `Torch.mul` Torch.asTensor (1 - eps :: Float)) `Torch.add` Torch.asTensor (eps :: Float)

-- | @(1 - eps) x@: pull degrees away from 1.
clampNotOne :: Float -> Torch.Tensor -> Torch.Tensor
clampNotOne eps x = x `Torch.mul` Torch.asTensor (1 - eps :: Float)
