{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

-- | A model INTERPRETATION: a LeNet-style MNIST CNN as an instance of the 'Model'
--   signature. [B,1,28,28] -> [B,10] raw logits. Carries its own parameter space
--   ('CNNSpace'), the forward body, and the initializer ('newCNN').
--   Shapes: 28 -conv5-> 24 -pool2-> 12 -conv5-> 8 -pool2-> 4, so flatten = 16*4*4 = 256.
module C_Domain.Models.Interpretations.MnistCNN (CNNSpace (..), newCNN) where

import C_Domain.Models.Signature (Model (..))
import GHC.Generics (Generic)
import Torch (Conv2d, Conv2dSpec (..), Linear, LinearSpec (..), Parameterized, Randomizable (..), conv2dForward, linear)
import qualified Torch
import qualified Torch.Functional as F

data CNNSpace = CNNSpace
  { conv1 :: Conv2d, -- 1 -> 6, kernel 5x5
    conv2 :: Conv2d, -- 6 -> 16, kernel 5x5
    fc1 :: Linear,   -- 256 -> 100
    fc2 :: Linear    -- 100 -> 10
  }
  deriving (Generic, Show, Parameterized)

instance Model CNNSpace where
  forward CNNSpace {..} x =
    let pool t = F.maxPool2d (2, 2) (2, 2) (0, 0) (1, 1) F.Floor t
        h1 = pool (F.relu (conv2dForward conv1 (1, 1) (0, 0) x)) -- [B,6,12,12]
        h2 = pool (F.relu (conv2dForward conv2 (1, 1) (0, 0) h1)) -- [B,16,4,4]
        flat = Torch.reshape [head (Torch.shape x), 256] h2 -- [B,256]
        d1 = F.elu (1.0 :: Float) (linear fc1 flat)
     in linear fc2 d1 -- [B,10] raw logits

-- | Draw fresh random weights (fixed LeNet architecture for 28x28 -> 10).
newCNN :: IO CNNSpace
newCNN =
  CNNSpace
    <$> sample (Conv2dSpec 1 6 5 5)
    <*> sample (Conv2dSpec 6 16 5 5)
    <*> sample (LinearSpec 256 100)
    <*> sample (LinearSpec 100 10)
