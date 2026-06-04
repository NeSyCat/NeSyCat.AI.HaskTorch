{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Interpretation I_gamma for Binary classification. The sorts are universe-invariant plain
--   types (a point is a tensor in both universes), so there is no per-universe sort assignment
--   and no @encPoint@ bridge. The only per-universe content is the two Kleisli relations:
--
--     classifierA \@GeomU = LogLeaf [True,False] . mlp theta   -- raw logits over {True,False}
--     classifierA \@MeasU = decode . classifierA \@GeomU theta  -- the MeasU reading is decode of that
--     labelA      = the circle test as a CERTAIN distribution (delta) -- 'encode' of an observation
module Binary.C_Domain.Interpretation
  ( module Binary.C_Domain.Signature,
    Params,
    initParams,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU, MeasU)
import A_Categorical.Category.Bridge (decode)
import A_Categorical.Category.Monads.Dist ()
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import B_Logical.Library.Stable (clampNotZero)
import Binary.C_Domain.Signature (BinaryKlRel (..), BinaryRel (..))
import C_Domain.NeuralNets.MLP (mlp, mlpArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)
import qualified Torch
import qualified Torch.Functional.Internal as F

-- | The parameter space: the pure weights of the MLP.
type Params = Weights

-- | Draw the initial theta_0 — fresh weights for the MLP.
initParams :: IO Params
initParams = sampleWeights mlpArch

-- | The ground-truth circle membership: inside the disc of radius^2 0.09 about (0.5, 0.5).
--   Operates on a single point as a @[2]@ tensor (MeasU's per-point view).
circleTest :: Torch.Tensor -> Bool
circleTest pt =
  let [x1, x2] = Torch.asValue pt :: [Float]
      dx = x1 - 0.5
      dy = x2 - 0.5
   in dx * dx + dy * dy < 0.09

instance BinaryRel MeasU where
  labelA pt = pure (circleTest pt) -- a certain distribution on the ground-truth label

instance BinaryRel GeomU where
  -- the label as a batched CERTAIN distribution: a one-hot delta per batch row (the GeomU
  -- analogue of MNIST's 'obsLeaf'); operates on the whole [B,2] batch.
  labelA pt =
    let center = F.mulScalar (Torch.onesLike pt) (0.5 :: Float)
        diff = pt `Torch.sub` center
        dist2 = Torch.sumDim (Torch.Dim (-1)) Torch.KeepDim Torch.Float (diff * diff) -- [B,1]
        radiusSq = F.mulScalar (Torch.onesLike dist2) (0.09 :: Float)
        insideF = Torch.toType Torch.Float (Torch.lt dist2 radiusSq) -- [B,1] in {0,1}
        oneHot = Torch.cat (Torch.Dim 1) [insideF, Torch.onesLike insideF `Torch.sub` insideF] -- [B,2]
     in LogLeaf [True, False] (Torch.log (clampNotZero 1e-13 oneHot))

instance BinaryKlRel MeasU where
  -- the MeasU reading IS 'decode' of the GeomU leaf (a point is the same tensor; reshape the
  -- single [2] point to [1,2] for the net).
  classifierA theta pt = decode (classifierA @GeomU theta (Torch.reshape [1, 2] pt))

instance BinaryKlRel GeomU where
  -- the classifier as a Bernoulli over {True,False}: the net's two raw logits as a LogVec leaf.
  classifierA theta = LogLeaf [True, False] . mlp theta -- [B,2] raw logits
