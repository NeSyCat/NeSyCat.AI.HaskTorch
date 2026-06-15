-- | MNIST CNN ARCHITECTURES + their forwards. Two LeNet-style digit classifiers, one matching
--   each reference system's net. BOTH output RAW logits — NeSyCat keeps the softmax OUT of the
--   net; it is supplied implicitly by the logsumexp normalizer at marginalization:
--
--     * 'cnnArch' / 'cnn'        — LTN-matched: head 256 -> 100 -> 84 -> 10, ELU
--                                  (matches LTN's @SingleDigit@ net).
--     * 'cnnArchDPL' / 'cnnDPL'  — DeepProbLog-matched: head 256 -> 120 -> 84 -> 10, ReLU
--                                  (DPL's @MNIST_Net@ head, but WITHOUT its in-net @Softmax@).
--
--   Shared LeNet body: 28 -conv5-> 24 -pool2-> 12 -conv5-> 8 -pool2-> 4, flatten 16*4*4 = 256.
--   @cnn = runArch cnnArch@ so a call site writes @cnn θ x@. Draw θ with @sampleWeights cnnArch@
--   (resp. @cnnArchDPL@).
module C_Domain.NeuralNets.MnistCNN (cnnArch, cnn, cnnArchDPL, cnnDPL) where

import C_Domain.NeuralNets.DSL.Semantics (Weights, runArch)
import C_Domain.NeuralNets.DSL.Syntax (Arch, Layer (..), (>>>))
import Torch (Tensor)

-- | LTN-matched MNIST CNN: [B,1,28,28] -> [B,10] raw logits. Head 100, ELU (conv: Conv->ELU->Pool).
cnnArch :: Arch
cnnArch =
  convBlock 1 6 >>> convBlock 6 16 >>> [Flatten, Linear 256 100, ELU, Linear 100 84, ELU, Linear 84 10]
  where
    convBlock i o = [Conv2d i o 5, ELU, MaxPool]

-- | The LTN-matched forward at θ: @cnn θ x = runArch cnnArch θ x@.
cnn :: Weights -> Tensor -> Tensor
cnn = runArch cnnArch

-- | DeepProbLog-matched MNIST CNN: [B,1,28,28] -> [B,10] RAW logits (DPL's @MNIST_Net@ minus the
--   in-net @Softmax@). Head 120, ReLU; conv blocks Conv->Pool->ReLU (DPL's encoder order — ReLU
--   commutes with max-pool, so this equals Conv->ReLU->Pool).
cnnArchDPL :: Arch
cnnArchDPL =
  convBlock 1 6 >>> convBlock 6 16 >>> [Flatten, Linear 256 120, ReLU, Linear 120 84, ReLU, Linear 84 10]
  where
    convBlock i o = [Conv2d i o 5, MaxPool, ReLU]

-- | The DeepProbLog-matched forward at θ: @cnnDPL θ x = runArch cnnArchDPL θ x@.
cnnDPL :: Weights -> Tensor -> Tensor
cnnDPL = runArch cnnArchDPL
