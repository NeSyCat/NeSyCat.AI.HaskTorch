{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Interpretation I_gamma for MNIST single-digit addition, mirroring the binary
--   interpretation section-for-section: sort assignments, then the MeasU and
--   GeomU assignments of the symbols, then the bridge.
--
--     Sorts : I(Image), I(Digit)        per universe (MeasU, GeomU)
--     digit : cnn -- the chosen CNN architecture (= runArch cnnArch) at θ (pure weights)
--     bridge: encImage / decDigit between MeasU and GeomU
module MnistAddition.C_Domain.Interpretation
  ( module MnistAddition.C_Domain.Signature,
    Params,
    initParams,
    obsLeaf,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU, MeasU)
import A_Categorical.Category.Monads.Dist (Dist (..))
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import qualified B_Logical.Interpretations.Boolean as BoolLogic
import qualified B_Logical.Interpretations.TensorProb as TensProbLogic
import B_Logical.Library.Stable (clampNotZero)
import MnistAddition.C_Domain.Signature (MnistArith (..), MnistBridge (..), MnistKlRel (..), MnistParams (..), MnistSorts (..))
import C_Domain.NeuralNets.MnistCNN (cnn, cnnArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)
import qualified Torch
import qualified Torch.Functional as F

-- ============================================================
--  Sort assignments: I(Image), I(Digit)
-- ============================================================

instance MnistSorts MeasU where
  type Image MeasU = Torch.Tensor      -- a single image [1,28,28]
  type Digit MeasU = Int               -- a digit; the distribution is carried by Dist
  type Natural MeasU = Int             -- a natural number (the sum)
  type Omega MeasU = BoolLogic.Omega   -- = Bool (the logic's MeasU truth object)

instance MnistSorts GeomU where
  type Image GeomU = Torch.Tensor          -- a batch [B,1,28,28]
  type Digit GeomU = Int                   -- a digit index 0..9 (the batch lives in the LogVec weights)
  type Natural GeomU = Int                 -- a sum index 0..18 (likewise)
  type Omega GeomU = TensProbLogic.OmegaP  -- = [0,1] truth (the fuzzy/product logic of TensorProb)

-- ============================================================
--  Parameter spaces (horizontal sorts): I(ThetaCNN) = the pure weights of 'arch'
-- ============================================================

-- | The horizontal sort: the PURE parameters (weights) of 'cnnArch'.
type Params = Weights

instance MnistParams MeasU where type ThetaCNN MeasU = Weights

instance MnistParams GeomU where type ThetaCNN GeomU = Weights

-- | Draw the initial theta_0 — fresh weights for 'cnnArch'.
initParams :: IO Params
initParams = sampleWeights cnnArch

-- ============================================================
--  MeasU: relation & function symbols (digit, +, =)
-- ============================================================

instance MnistKlRel MeasU where
  digit :: Weights -> Image MeasU -> Dist (Digit MeasU)
  digit theta img =
    let logits = cnn theta (encImage @MeasU @GeomU img)        -- run the chosen architecture at θ (logits)
        ps = head (Torch.asValue (F.softmax (F.Dim 1) logits) :: [[Float]])
     in FiniteSupp [(d, realToFrac (ps !! d)) | d <- [0 .. 9]] -- softmax -> Dist over 0..9

instance MnistArith MeasU where
  -- (+) : host integer addition. The Sigma (law of total probability) is supplied
  --       by the Dist bind in the formula, not here.
  plus :: Int -> Int -> Int
  plus = (+)

  -- (=) : ordinary equality of naturals -> Bool.
  eqNat :: Int -> Int -> Bool
  eqNat = (==)

-- ============================================================
--  GeomU: relation & function symbols (digit, +, =)
-- ============================================================

instance MnistKlRel GeomU where
  digit :: Weights -> Image GeomU -> LogVec (Digit GeomU)
  digit theta img = LogLeaf [0 .. 9] (cnn theta img) -- the batched digit distribution as raw logits

instance MnistArith GeomU where
  -- (+) : literally host integer addition of the two digit indices. The convolution
  --       (LogSumExp over the anti-diagonal d1+d2=s) is supplied by the LogVec BIND,
  --       not here -- exactly as MeasU's (+) leans on the Dist bind.
  plus :: Int -> Int -> Int
  plus = (+)

  -- (=) : crisp equality of two sum indices, as the (degenerate) fuzzy truth 0/1. The
  --       per-(d1,d2,s) truths are marginalized to a probability by the LogVec readout
  --       (logVecReadoutP), reproducing the old softmax-of-logConv exactly.
  eqNat :: Int -> Int -> TensProbLogic.OmegaP
  eqNat a b = TensProbLogic.OmegaP (Torch.asTensor [if a == b then 1.0 else 0.0 :: Float])

-- ============================================================
--  BRIDGE: MeasU <-> GeomU (softmax decode to a Dist over digits)
-- ============================================================

instance MnistBridge MeasU GeomU where
  -- An image is the same pixel tensor in both universes (no representation gap,
  -- unlike binary's (Float,Float) -> tensor), so the encode is the identity.
  encImage :: Image MeasU -> Image GeomU
  encImage img = img

-- | Wrap the observed one-hot sum @[B,19]@ into a (certain) 'LogVec' distribution over
--   the sum indices @0..18@ -- a batched point mass. Lets the observed sum flow through
--   the monad just like the digit predictions, so the formula stays universe-uniform.
obsLeaf :: Torch.Tensor -> LogVec Int
obsLeaf oneHotN = LogLeaf [0 .. 18] (Torch.log (clampNotZero 1e-13 oneHotN))
