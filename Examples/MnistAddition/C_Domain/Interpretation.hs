{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}

-- | Interpretation I_gamma for MNIST single-digit addition. The sorts are universe-invariant
--   plain types (see "MnistAddition.C_Domain.Signature"), so there is no per-universe sort
--   assignment and no image bridge -- an image is the same tensor in both readings. The only
--   per-universe content is 'digit':
--
--     digit \@GeomU = LogLeaf [0..9] . cnn theta   -- raw logits as a LogVec leaf
--     digit \@MeasU = decode . digit \@GeomU theta  -- the MeasU reading is decode of that leaf
module MnistAddition.C_Domain.Interpretation
  ( module MnistAddition.C_Domain.Signature,
    Params,
    initParams,
    obsLeaf,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU, MeasU)
import A_Categorical.Category.Bridge (decode)
import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import B_Logical.Library.Stable (clampNotZero)
import MnistAddition.C_Domain.Signature (Digit, Image, MnistKlRel (..), Natural)
import C_Domain.NeuralNets.MnistCNN (cnn, cnnArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)
import qualified Torch

-- | The parameter space: the pure weights of 'cnnArch'.
type Params = Weights

-- | Draw the initial theta_0 — fresh weights for 'cnnArch'.
initParams :: IO Params
initParams = sampleWeights cnnArch

instance MnistKlRel GeomU where
  digit :: Weights -> Image -> LogVec Digit
  digit theta img = LogLeaf [0 .. 9] (cnn theta img) -- raw logits over 0..9 (no softmax)

instance MnistKlRel MeasU where
  digit :: Weights -> Image -> Dist Digit
  digit theta = decode . digit @GeomU theta
    -- the MeasU reading IS 'decode' of the GeomU leaf (an image is the same tensor; no bridge)

-- | The observed one-hot sum @[B,19]@ as a CERTAIN LogVec over 0..18 (a batched point mass) --
--   the 'encode' of the observation, realized at the construction site (support 0..18 explicit).
obsLeaf :: Torch.Tensor -> LogVec Natural
obsLeaf oneHotN = LogLeaf [0 .. 18] (Torch.log (clampNotZero 1e-13 oneHotN))
