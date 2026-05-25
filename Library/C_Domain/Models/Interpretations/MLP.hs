{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

-- | A model INTERPRETATION: a small two-hidden-layer MLP as an instance of the
--   'Model' signature. All concrete (mirroring "B_Logical.Interpretations.*") — it
--   carries its own parameter space ('MLPSpace'), the forward body, and the
--   initializer ('newMLP'). Generic in its dimensions, so any example can reuse it.
module C_Domain.Models.Interpretations.MLP (MLPSpace (..), newMLP) where

import C_Domain.Models.Signature (Model (..))
import GHC.Generics (Generic)
import Torch (Linear, LinearSpec (..), Parameterized, Randomizable (..), linear)
import qualified Torch.Functional as F

-- | The parameter space (the weights). 'Parameterized' so the optimizer can
--   flatten/replace its tensors.
data MLPSpace = MLPSpace
  { fc1 :: Linear,
    fc2 :: Linear,
    fc3 :: Linear
  }
  deriving (Generic, Show, Parameterized)

instance Model MLPSpace where
  forward MLPSpace {..} input =
    let l1 = F.elu (1.0 :: Float) (linear fc1 input)
        l2 = F.elu (1.0 :: Float) (linear fc2 l1)
     in linear fc3 l2

-- | Draw fresh random weights for the given dims (inDim, hiddenDim, outDim).
newMLP :: Int -> Int -> Int -> IO MLPSpace
newMLP inDim hiddenDim outDim =
  MLPSpace
    <$> sample (LinearSpec inDim hiddenDim)
    <*> sample (LinearSpec hiddenDim hiddenDim)
    <*> sample (LinearSpec hiddenDim outDim)
