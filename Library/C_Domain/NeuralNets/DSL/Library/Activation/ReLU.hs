-- | Activation (parameter space ℝ⁰): the ReLU nonlinearity.
module C_Domain.NeuralNets.DSL.Library.Activation.ReLU (relu) where

import Torch (Tensor)
import qualified Torch.Functional as F

relu :: Tensor -> Tensor
relu = F.relu
