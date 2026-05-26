-- | Parameterized (parameter space ℝ^{i·o·k²+o}): a 2-D convolution (stride 1, no
--   padding). 'sampleConv2d' draws fresh θ; 'conv2d' is the forward @θ -> Tensor -> Tensor@.
module C_Domain.Library.Parameterized.Conv2d (sampleConv2d, conv2d) where

import Torch (Conv2d, Conv2dSpec (..), Tensor, sample)
import qualified Torch

sampleConv2d :: Int -> Int -> Int -> IO Conv2d
sampleConv2d i o k = sample (Conv2dSpec i o k k)

conv2d :: Conv2d -> Tensor -> Tensor
conv2d c = Torch.conv2dForward c (1, 1) (0, 0)
