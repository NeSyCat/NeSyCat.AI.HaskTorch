-- | The MNIST CNN ARCHITECTURE, plus its forward. 'cnnArch' is the pure 'Arch'
--   (the single source of truth); 'cnn' is its
--   forward at θ — @cnn = runArch cnnArch@, so a call site writes @cnn θ x@ instead of
--   @runArch cnnArch θ x@. Draw θ with @sampleWeights cnnArch@.
--   LeNet-style: 28 -conv5-> 24 -pool2-> 12 -conv5-> 8 -pool2-> 4, flatten 16*4*4 = 256.
module C_Domain.Architecture.MnistCNN (cnnArch, cnn) where

import C_Domain.Interpretation (Weights, runArch)
import C_Domain.Signature (Arch, Layer (..), (>>>))
import Torch (Tensor)

-- | LeNet-style MNIST CNN: [B,1,28,28] -> [B,10] raw logits.
cnnArch :: Arch
cnnArch =
  convBlock 1 6 >>> convBlock 6 16 >>> [Flatten, Linear 256 100, ELU, Linear 100 10]
  where
    convBlock i o = [Conv2d i o 5, ReLU, MaxPool]

-- | The CNN forward at θ: @cnn θ x = runArch cnnArch θ x@.
cnn :: Weights -> Tensor -> Tensor
cnn = runArch cnnArch
