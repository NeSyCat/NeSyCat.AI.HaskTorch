-- | Truth type for real-valued (tensor) logic: Omega = R^1 in logit space.
--
-- Values are unconstrained reals (logits), not probabilities.
-- +inf encodes certainty of truth, -inf certainty of falsehood.
-- This corresponds to the standard model over (R, +, ×, ≤).
module B_Logical.Truth.Omega.Tensor (Omega) where

import qualified Torch

-- | Omega := R^1 represented as a rank-1 Float tensor of shape [1].
type Omega = Torch.Tensor
