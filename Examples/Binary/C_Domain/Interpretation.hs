{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Interpretation I_gamma for Binary Classification, built like MnistAddition (the
--   probabilistic reading): the truth object is a CRISP @Bool@ in BOTH universes, and the
--   satisfaction degree rides in the monad (a @Dist Bool@ / @LogVec Bool@), not in the truth
--   value. So Binary now shares MNIST's truth algebra and quantifier; the only genuinely
--   per-universe symbols are the two Kleisli relations:
--
--     classifierA : the MLP as a Bernoulli over the label -- @Dist Bool@ (MeasU) /
--                   @LogVec Bool@ leaf @LogLeaf [True,False] [logit,0]@ (GeomU); P(True)=sigmoid(logit).
--     labelA      : the ground-truth circle test as a CERTAIN (delta) distribution -- the
--                   analogue of MNIST's observed-sum leaf, so the batch rides in the weights.
--
--   No @decOmega@ bridge: each universe builds its own distribution. @encPoint@ stays (the
--   @Point@ representations differ: a host tuple in MeasU, a tensor in GeomU).
module Binary.C_Domain.Interpretation
  ( module Binary.C_Domain.Signature,
    Params,
    initParams,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU, MeasU)
import A_Categorical.CategoricalSignature (Framework)
import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.Category.Monads.DistDecode (categorical)
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import B_Logical.Library.Stable (clampNotZero)
import Binary.C_Domain.Signature (BinaryBridge (..), BinaryKlRel (..), BinaryParams (..), BinaryRel (..), BinarySorts (..))
import C_Domain.NeuralNets.MLP (mlp, mlpArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)
import qualified Torch
import qualified Torch.Functional.Internal as F

-- ============================================================
--  Sort assignments: I(Point), I(Omega)
-- ============================================================

instance BinarySorts MeasU where
  type Point MeasU = (Float, Float)   -- R^2 as a Cartesian product
  type Omega MeasU = Bool             -- the shared crisp truth object (Dist carries the degree)

instance BinarySorts GeomU where
  type Point GeomU = Torch.Tensor     -- a batch [B,2], dtype Float
  type Omega GeomU = Bool             -- the SAME crisp truth object as MeasU (LogVec carries the degree)

-- ============================================================
--  Parameter spaces (horizontal sorts): I(Theta) = the pure weights of the MLP
-- ============================================================

-- | The horizontal sort: the PURE parameters (weights). Binary's architecture is the
--   imported 'mlpArch' (chosen here, in this interpretation); only θ varies, and the
--   forward is 'mlp' (= @runArch mlpArch@).
type Params = Weights

-- Universe-invariant: the MLP weight space is the same in every interpretation, so ONE
-- polymorphic instance covers all universes.
instance (Framework u) => BinaryParams u where type Theta u = Weights

-- | Draw the initial theta_0 — fresh weights for the MLP.
initParams :: IO Params
initParams = sampleWeights mlpArch

-- ============================================================
--  MeasU: the label (certain) + the classifier (Bernoulli), both as Dist Bool
-- ============================================================

instance BinaryRel MeasU where
  labelA :: Point MeasU -> Dist (Omega MeasU)
  labelA pt = pure (circleTest pt)              -- a certain distribution on the ground-truth label

instance BinaryKlRel MeasU where
  classifierA :: Weights -> Point MeasU -> Dist (Omega MeasU)
  classifierA theta pt =
    -- pure composition: encode the point, run the net (2 logits), the modular softmax decode
    categorical [True, False] (mlp theta (Torch.reshape [1, 2] (encPoint @MeasU @GeomU pt)))

-- | The ground-truth circle membership: inside the disc of radius^2 0.09 about (0.5, 0.5).
circleTest :: (Float, Float) -> Bool
circleTest (x1, x2) =
  let dx = x1 - 0.5
      dy = x2 - 0.5
   in dx * dx + dy * dy < 0.09

-- ============================================================
--  GeomU: the label (certain delta leaf) + the classifier (Bernoulli leaf), as LogVec Bool
-- ============================================================

instance BinaryRel GeomU where
  -- | The label as a batched CERTAIN distribution: a one-hot delta on the true label per
  --   batch row (the GeomU analogue of MNIST's observed-sum leaf 'obsLeaf').
  labelA :: Point GeomU -> LogVec (Omega GeomU)
  labelA pt =
    let center = F.mulScalar (Torch.onesLike pt) (0.5 :: Float)
        diff = pt `Torch.sub` center
        dist2 = Torch.sumDim (Torch.Dim (-1)) Torch.KeepDim Torch.Float (diff * diff) -- [B,1]
        radiusSq = F.mulScalar (Torch.onesLike dist2) (0.09 :: Float)
        insideF = Torch.toType Torch.Float (Torch.lt dist2 radiusSq)                  -- [B,1] in {0,1}
        oneHot = Torch.cat (Torch.Dim 1) [insideF, Torch.onesLike insideF `Torch.sub` insideF] -- [B,2]
     in LogLeaf [True, False] (Torch.log (clampNotZero 1e-13 oneHot))

instance BinaryKlRel GeomU where
  -- | The classifier as a Bernoulli over the label: the net's two raw logits as a LogVec leaf
  --   (pure logits; @P(True) = softmax(.)[0]@ is the implicit log-space readout, never here).
  classifierA :: Weights -> Point GeomU -> LogVec (Omega GeomU)
  classifierA theta = LogLeaf [True, False] . mlp theta -- [B,2] raw logits

-- ============================================================
--  BRIDGE: MeasU <-> GeomU (Point representation only)
-- ============================================================

instance BinaryBridge MeasU GeomU where
  encPoint :: Point MeasU -> Point GeomU
  encPoint (x1, x2) =
    Torch.toDevice (Torch.Device Torch.CPU 0) (Torch.asTensor [x1, x2])
