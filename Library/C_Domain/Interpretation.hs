{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

-- | The architecture DSL — INTERPRETATIONS of the abstract 'Layer' vocabulary.
--   Crucially, the ARCHITECTURE and the PARAMETERS are kept separate:
--
--     * 'Weights' is θ — the PURE parameters (just the sampled weights, no forwards,
--       no architecture). 'Parameterized', so the optimizer trains it.
--     * 'sampleWeights' draws fresh θ for an architecture; 'runArch' applies a FIXED
--       architecture to θ. So the axiom sees only θ ('Weights'), and the architecture
--       is supplied separately (by whoever holds it — the example's C interpretation).
--
--   'sampleWeights' and 'runArch' are two interpretations of the same pure 'Arch'.
--   To add a learnable layer: add its 'Layer' symbol + a 'LayerWeight' constructor
--   here + a case in 'sampleWeights' and 'runArch'.
module C_Domain.Interpretation
  ( Weights,
    sampleWeights,
    runArch,
  )
where

import C_Domain.Signature (Arch, Layer (..))
import Data.Maybe (catMaybes)
import GHC.Generics (Generic)
import Torch (Conv2d, Conv2dSpec (..), Linear, LinearSpec (..), Parameterized (..), Randomizable (..), Tensor, conv2dForward, linear)
import qualified Torch
import qualified Torch.Functional as F

-- | The sampled weights of one learnable layer (one constructor per learnable kind).
data LayerWeight = LinearW Linear | Conv2dW Conv2d
  deriving (Generic, Show, Parameterized)

-- | θ — the PARAMETER SPACE: just the sampled weights, in architecture order. No
--   forwards, no architecture. (The architecture is applied separately by 'runArch'.)
newtype Weights = Weights [LayerWeight]

instance Parameterized Weights where
  flattenParameters (Weights ws) = concatMap flattenParameters ws
  _replaceParameters (Weights ws) = Weights <$> mapM _replaceParameters ws

-- | Draw fresh θ for an architecture — sample each learnable layer (in order).
sampleWeights :: Arch -> IO Weights
sampleWeights arch = (Weights . catMaybes) <$> mapM sampleLayer arch
  where
    sampleLayer (Linear i o) = Just . LinearW <$> sample (LinearSpec i o)
    sampleLayer (Conv2d i o k) = Just . Conv2dW <$> sample (Conv2dSpec i o k k)
    sampleLayer _ = pure Nothing -- parameter-free layers carry no weights

-- | Apply a FIXED architecture to θ on an input: walk the layers, pulling a weight
--   for each learnable layer (in order) and applying parameter-free layers directly.
runArch :: Arch -> Weights -> Tensor -> Tensor
runArch arch (Weights ws0) = go arch ws0
  where
    go (Linear _ _ : ls) (LinearW l : ws) x = go ls ws (linear l x)
    go (Conv2d _ _ _ : ls) (Conv2dW c : ws) x = go ls ws (conv2dForward c (1, 1) (0, 0) x)
    go (ELU : ls) ws x = go ls ws (F.elu (1.0 :: Float) x)
    go (ReLU : ls) ws x = go ls ws (F.relu x)
    go (Sigmoid : ls) ws x = go ls ws (Torch.sigmoid x)
    go (MaxPool : ls) ws x = go ls ws (F.maxPool2d (2, 2) (2, 2) (0, 0) (1, 1) F.Floor x)
    go (Flatten : ls) ws x = go ls ws (Torch.reshape [head (Torch.shape x), product (tail (Torch.shape x))] x)
    go _ _ x = x -- empty arch, or a weights/arch mismatch (untyped: no compile-time guard)
