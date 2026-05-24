{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

-- | A LeNet-style MNIST CNN, mirroring the untyped 'C_Domain.Models.MLP' pattern.
--   Forward: [B,1,28,28] -> [B,10] raw logits. Like @hThetaReal@ for the MLP, the
--   model is logits-only; turning logits into a digit *distribution* (softmax) is
--   the bridge, and lives in the interpretation (cf. the binary @decOmega@).
--   Shapes: 28 -conv5-> 24 -pool2-> 12 -conv5-> 8 -pool2-> 4, so flatten = 16*4*4 = 256.
module C_Domain.Models.MnistCNN
  ( ParamsCNN (..),
    ParamsCNNSpec (..),
    cnnLogits,
  )
where

import GHC.Generics (Generic)
import Torch
  ( Conv2d,
    Conv2dSpec (..),
    Linear,
    LinearSpec (..),
    Parameterized,
    Randomizable (..),
    Tensor,
    conv2dForward,
    linear,
  )
import qualified Torch
import qualified Torch.Functional as F

data ParamsCNNSpec = ParamsCNNSpec deriving (Show, Eq)

data ParamsCNN = ParamsCNN
  { conv1 :: Conv2d, -- 1 -> 6, kernel 5x5
    conv2 :: Conv2d, -- 6 -> 16, kernel 5x5
    fc1 :: Linear,   -- 256 -> 100
    fc2 :: Linear    -- 100 -> 10
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable ParamsCNNSpec ParamsCNN where
  sample ParamsCNNSpec =
    ParamsCNN
      <$> sample (Conv2dSpec 1 6 5 5)
      <*> sample (Conv2dSpec 6 16 5 5)
      <*> sample (LinearSpec 256 100)
      <*> sample (LinearSpec 100 10)

-- | Forward pass: [B,1,28,28] -> [B,10] logits.
cnnLogits :: ParamsCNN -> Tensor -> Tensor
cnnLogits ParamsCNN {..} x =
  let pool t = F.maxPool2d (2, 2) (2, 2) (0, 0) (1, 1) F.Floor t
      h1 = pool (F.relu (conv2dForward conv1 (1, 1) (0, 0) x)) -- [B,6,12,12]
      h2 = pool (F.relu (conv2dForward conv2 (1, 1) (0, 0) h1)) -- [B,16,4,4]
      flat = Torch.reshape [head (Torch.shape x), 256] h2 -- [B,256]
      d1 = F.elu (1.0 :: Float) (linear fc1 flat)
   in linear fc2 d1 -- [B,10] raw logits
