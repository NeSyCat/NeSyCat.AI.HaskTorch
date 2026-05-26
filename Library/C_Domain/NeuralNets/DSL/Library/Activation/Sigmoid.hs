-- | Activation (parameter space ℝ⁰): the logistic sigmoid.
module C_Domain.NeuralNets.DSL.Library.Activation.Sigmoid (sigmoid) where

import Torch (Tensor)
import qualified Torch

sigmoid :: Tensor -> Tensor
sigmoid = Torch.sigmoid
