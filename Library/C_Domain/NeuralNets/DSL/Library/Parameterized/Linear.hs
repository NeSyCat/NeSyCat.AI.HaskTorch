-- | Parameterized (parameter space ℝ^{i·o+o}): an affine map. 'sampleLinear' draws
--   fresh θ for it; 'linear' is the forward @θ -> Tensor -> Tensor@.
module C_Domain.NeuralNets.DSL.Library.Parameterized.Linear (sampleLinear, linear) where

import Torch (Linear, LinearSpec (..), Tensor, sample)
import qualified Torch

sampleLinear :: Int -> Int -> IO Linear
sampleLinear i o = sample (LinearSpec i o)

linear :: Linear -> Tensor -> Tensor
linear = Torch.linear
