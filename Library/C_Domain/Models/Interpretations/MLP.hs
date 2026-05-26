{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}

-- | A model INTERPRETATION: a small two-hidden-layer MLP as an instance of the
--   'Model' signature. Everything is in the instance — the param space ('MLPSpace'),
--   'forward' (the map), and 'fresh' (init from @(inDim, hiddenDim, outDim)@) —
--   mirroring "B_Logical.Interpretations.*". Generic in its dimensions.
module C_Domain.Models.Interpretations.MLP (MLPSpace (..)) where

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
  type Init MLPSpace = (Int, Int, Int) -- (inDim, hiddenDim, outDim)

  forward MLPSpace {..} input =
    let l1 = F.elu (1.0 :: Float) (linear fc1 input)
        l2 = F.elu (1.0 :: Float) (linear fc2 l1)
     in linear fc3 l2

  fresh (inDim, hiddenDim, outDim) =
    MLPSpace
      <$> sample (LinearSpec inDim hiddenDim)
      <*> sample (LinearSpec hiddenDim hiddenDim)
      <*> sample (LinearSpec hiddenDim outDim)
