-- | Entailment order on tensor truth values: vdash = (<=) on extracted Float.
--
-- Compares two Omega tensors and returns a meta-truth (Bool).
-- a |= b iff the scalar value of a is <= the scalar value of b.
module B_Logical.Comparator.Vdash.Tensor (vdash) where

import qualified Torch

-- | Tensor entailment: compare scalar Float values of rank-1 tensors.
vdash :: Torch.Tensor -> Torch.Tensor -> Bool
vdash a b = Torch.asValue a <= (Torch.asValue b :: Float)
