-- | A model INTERPRETATION: the MNIST CNN, as a Sequential ARCHITECTURE plus its
--   forward. 'cnnArch' is the pure 'Arch' (the single source of truth); 'cnn' is its
--   forward at θ — @cnn = runArch cnnArch@, so a call site writes @cnn θ x@ instead of
--   @runArch cnnArch θ x@. Draw θ with @sampleWeights cnnArch@.
--   LeNet-style: 28 -conv5-> 24 -pool2-> 12 -conv5-> 8 -pool2-> 4, flatten 16*4*4 = 256.
module C_Domain.Models.Interpretations.MnistCNN (cnnArch, cnn) where

import C_Domain.Models.Sequential.Interpretation (Weights, runArch)
import C_Domain.Models.Sequential.Signature (Arch, conv2dL, eluL, flattenL, linearL, maxPoolL, reluL, (>>>))
import Torch (Tensor)

-- | LeNet-style MNIST CNN: [B,1,28,28] -> [B,10] raw logits.
cnnArch :: Arch
cnnArch =
  conv2dL 1 6 5 >>> reluL >>> maxPoolL
    >>> conv2dL 6 16 5 >>> reluL >>> maxPoolL
    >>> flattenL
    >>> linearL 256 100 >>> eluL
    >>> linearL 100 10

-- | The CNN forward at θ: @cnn θ x = runArch cnnArch θ x@.
cnn :: Weights -> Tensor -> Tensor
cnn = runArch cnnArch
