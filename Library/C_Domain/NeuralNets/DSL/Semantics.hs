{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

-- | The architecture DSL — INTERPRETATIONS of the abstract 'Layer' vocabulary.
--   Architecture and parameters are kept separate: 'Weights' is θ (the pure params);
--   the architecture is supplied separately by whoever holds it.
--
--   The interpretation is given as TWO explicit per-symbol TABLES — one assignment
--   line per 'Layer' symbol, exactly like a logical signature's symbols get their
--   Boolean / Tensor meanings:
--
--     * 'sampleLayer' — symbol ↦ how to draw its weight   (the PARAMETER reading)
--     * 'runLayer'    — symbol ↦ its tensor function       (the FORWARD reading)
--
--   'sampleWeights' / 'runArch' are then GENERIC FOLDS over those tables (collect /
--   thread the weights, compose) — they carry no per-symbol knowledge themselves.
--   Each symbol's function lives in "C_Domain.NeuralNets.DSL.Library.*"; nothing here inlines
--   @Torch.*@. To add a learnable layer: a 'Layer' symbol + a 'LayerWeight'
--   constructor + a "C_Domain.NeuralNets.DSL.Library.*" function + one line in each table.
module C_Domain.NeuralNets.DSL.Semantics
  ( Weights,
    sampleWeights,
    runArch,
    segmentWeights,
  )
where

import C_Domain.NeuralNets.DSL.Library.Activation.ELU (elu)
import C_Domain.NeuralNets.DSL.Library.Activation.ReLU (relu)
import C_Domain.NeuralNets.DSL.Library.Activation.Sigmoid (sigmoid)
import C_Domain.NeuralNets.DSL.Library.Parameterized.BiGRU (BiGRUW, bigru, sampleBiGRU)
import C_Domain.NeuralNets.DSL.Library.Parameterized.Conv2d (conv2d, sampleConv2d)
import C_Domain.NeuralNets.DSL.Library.Parameterized.Embedding (EmbedW, embed, sampleEmbed)
import C_Domain.NeuralNets.DSL.Library.Parameterized.Linear (linear, sampleLinear)
import C_Domain.NeuralNets.DSL.Library.Shape.Flatten (flatten)
import C_Domain.NeuralNets.DSL.Library.Shape.MaxPool (maxPool)
import C_Domain.NeuralNets.DSL.Syntax (Arch, Layer (..))
import Data.List (foldl', mapAccumL)
import Data.Maybe (catMaybes, isJust)
import GHC.Generics (Generic)
import Torch (Conv2d, Linear, Parameterized (..), Tensor)

-- | The sampled weights of one learnable layer (one constructor per learnable kind).
data LayerWeight = LinearW Linear | Conv2dW Conv2d | EmbW EmbedW | GruW BiGRUW
  deriving (Generic, Show, Parameterized)

-- | θ — the PARAMETER SPACE: just the sampled weights, in architecture order.
newtype Weights = Weights [LayerWeight]

instance Parameterized Weights where
  flattenParameters (Weights ws) = concatMap flattenParameters ws
  _replaceParameters (Weights ws) = Weights <$> mapM _replaceParameters ws

-- ============================================================
--  The interpretation: each Layer SYMBOL ↦ its meaning (two readings)
-- ============================================================

-- | PARAMETER reading — each symbol ↦ how to draw its weight (Nothing = none).
sampleLayer :: Layer -> Maybe (IO LayerWeight)
sampleLayer (Linear i o) = Just (LinearW <$> sampleLinear i o)
sampleLayer (Conv2d i o k) = Just (Conv2dW <$> sampleConv2d i o k)
sampleLayer (Embedding v d) = Just (EmbW <$> sampleEmbed v d)
sampleLayer (BiGRU i h) = Just (GruW <$> sampleBiGRU i h)
sampleLayer _ = Nothing

-- | FORWARD reading — each symbol ↦ its tensor function, given the weight it consumes
--   (if any). Read it as: @Linear@ is interpreted as @linear@, @Conv2d@ as @conv2d@,
--   @ELU@ as @elu@, and so on.
runLayer :: Layer -> Maybe LayerWeight -> (Tensor -> Tensor)
runLayer (Linear _ _) (Just (LinearW l)) = linear l
runLayer (Conv2d _ _ _) (Just (Conv2dW c)) = conv2d c
runLayer (Embedding _ _) (Just (EmbW e)) = embed e
runLayer (BiGRU _ _) (Just (GruW g)) = bigru g
runLayer ELU _ = elu
runLayer ReLU _ = relu
runLayer Sigmoid _ = sigmoid
runLayer MaxPool _ = maxPool
runLayer Flatten _ = flatten
runLayer _ _ = id -- empty / weights-arch mismatch (untyped: no compile-time guard)

-- ============================================================
--  The folds: generic plumbing over the tables above (no per-symbol knowledge)
-- ============================================================

-- | Draw fresh θ — the PARAMETER reading collected over the architecture.
sampleWeights :: Arch -> IO Weights
sampleWeights = fmap (Weights . catMaybes) . traverse (sequenceA . sampleLayer)

-- | Run a FIXED architecture at θ — the FORWARD reading threaded over the architecture:
--   give each layer the next weight iff it is parameterized, then compose left to right.
runArch :: Arch -> Weights -> Tensor -> Tensor
runArch arch (Weights ws0) input = foldl' (flip ($)) input (snd (mapAccumL feed ws0 arch))
  where
    feed ws layer = case (sampleLayer layer, ws) of
      (Just _, w : rest) -> (rest, runLayer layer (Just w)) -- parameterized: take a weight
      _ -> (ws, runLayer layer Nothing) -- parameter-free (or mismatch): take none

-- | Split θ along a SEGMENTATION of an architecture (the segments, concatenated, are the
--   architecture θ was sampled from). Lets a NON-sequential model (a DAG: e.g. a shared
--   trunk feeding several heads, like the WAP encoder) be sampled as ONE 'Arch' (one θ for
--   the optimizer) and run piecewise with 'runArch' on its sequential segments.
segmentWeights :: [Arch] -> Weights -> [Weights]
segmentWeights segs (Weights ws0) = go segs ws0
  where
    go [] _ = []
    go (a : as) ws =
      let (w, rest) = splitAt (length (filter (isJust . sampleLayer) a)) ws
       in Weights w : go as rest
