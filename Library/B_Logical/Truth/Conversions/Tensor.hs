-- | Conversion functions into tensor truth (Omega = Torch.Tensor, logit space).
module B_Logical.Truth.Conversions.Tensor
  ( float2t
  , bool2t
  , logSigmoid
  , one
  , eps
  ) where

import qualified Torch

-- | Wrap a Float scalar as a rank-1 Omega tensor.
float2t :: Float -> Torch.Tensor
float2t x = Torch.asTensor [x]

-- | Lift a Bool into logit space: True -> +inf, False -> -inf.
bool2t :: Bool -> Torch.Tensor
bool2t True  = Torch.asTensor [(1.0 / 0.0) :: Float]
bool2t False = Torch.asTensor [(-1.0 / 0.0) :: Float]

-- | Numerically stable log-sigmoid: log σ(x) = -log(1 + exp(-x)).
-- Maps R -> (-inf, 0]: large positive -> 0, large negative -> -inf.
logSigmoid :: Torch.Tensor -> Torch.Tensor
logSigmoid x = negate (Torch.log (Torch.exp (negate x) `Torch.add` Torch.onesLike x))

-- | The scalar 1.0 (CPU Float tensor).
one :: Torch.Tensor
one = Torch.toDevice (Torch.Device Torch.CPU 0) $ Torch.asTensor (1.0 :: Float)

-- | A small epsilon for numerical stability.
eps :: Torch.Tensor
eps = Torch.toDevice (Torch.Device Torch.CPU 0) $ Torch.asTensor (1.0e-8 :: Float)
