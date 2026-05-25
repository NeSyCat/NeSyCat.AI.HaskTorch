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
--     digit : the CNN; GeomU = raw logits, MeasU = softmax -> Dist over 0..9 (via the bridge)
--     bridge: encImage / decDigit between MeasU and GeomU
module Examples.MnistAddition.C_Domain.Interpretation
  ( module Examples.MnistAddition.C_Domain.Signature,
    module Lib.Models.MnistCNN,
  )
where

import Lib.A_Categorical.CategoricalInterpretation (GeomU, MeasU)
import Lib.A_Categorical.Category.Monads.Dist (Dist (..))
import qualified Lib.B_Logical.Interpretations.Boolean as BoolLogic
import qualified Lib.B_Logical.Interpretations.Tensor as TensLogic
import Examples.MnistAddition.C_Domain.Signature (MnistArith (..), MnistBridge (..), MnistKlRel (..), MnistSorts (..))
import Lib.Models.MnistCNN (ParamsCNN, cnnLogits)
import Data.Functor.Identity (Identity (..))
import qualified Torch
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI

-- ============================================================
--  Sort assignments: I(Image), I(Digit)
-- ============================================================

instance MnistSorts MeasU where
  type Image MeasU = Torch.Tensor      -- a single image [1,28,28]
  type Digit MeasU = Int               -- a digit; the distribution is carried by Dist
  type Natural MeasU = Int             -- a natural number (the sum)
  type Omega MeasU = BoolLogic.Omega   -- = Bool (the logic's MeasU truth object)

instance MnistSorts GeomU where
  type Image GeomU = Torch.Tensor      -- a batch [B,1,28,28]
  type Digit GeomU = Torch.Tensor      -- logit vector [B,10]
  type Natural GeomU = Torch.Tensor    -- logit vector over sums [B,19] (one-hot for observed)
  type Omega GeomU = TensLogic.Omega   -- = Torch.Tensor (the logic's GeomU truth object)

-- ============================================================
--  MeasU: relation & function symbols (digit, +, =)
-- ============================================================

instance MnistKlRel MeasU where
  type ThetaCNN MeasU = ParamsCNN
  digit :: ParamsCNN -> Image MeasU -> Dist (Digit MeasU)
  digit theta img =
    let imgT = encImage @MeasU @GeomU img -- encode the image into a GeomU batch tensor
        logits = cnnLogits theta imgT      -- run the net (logits)
     in decDigit @MeasU @GeomU logits      -- decode logits -> Dist over 0..9

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
  type ThetaCNN GeomU = ParamsCNN
  digit :: ParamsCNN -> Image GeomU -> Identity (Digit GeomU)
  digit theta img = Identity (cnnLogits theta img) -- raw logits, like classifierA @GeomU

instance MnistArith GeomU where
  -- (+) : lifted addition of two digit-logit vectors [B,10] into a logit vector
  --       over sums [B,19] via LogSumExp over the anti-diagonal d1+d2=s. The (+)
  --       inside is @oplus@, the reduction is the logic's smooth sup -- both
  --       reused. On raw logits; no softmax, no normalization.
  plus :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
  plus = logConv

  -- (=) : pick out the observed sum's entry, @<one-hot n, logits>@ -> [B] truth.
  eqNat :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
  eqNat oneHotN sums =
    Torch.sumDim (Torch.Dim 1) Torch.RemoveDim Torch.Float (oneHotN `Torch.mul` sums)

-- | Log-space convolution of two digit-logit vectors, @[B,10] -> [B,10] -> [B,19]@:
--   @out[b,s] = LogSumExp_{i+j=s} (lx[b,i] + ly[b,j])@. The geometric (logit-space)
--   shadow of the law of total probability -- the same LogSumExp/add the TensReal
--   logic uses, none introduced.
logConv :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
logConv lx ly =
  let b = head (Torch.shape lx)
      jj = Torch.reshape [b, 10, 1] lx `Torch.add` Torch.reshape [b, 1, 10] ly -- [B,10,10]
      entry i j = FI.select (FI.select jj 1 i) 1 j -- jj[:,i,j] :: [B]
      colFor s =
        let es = [entry i (s - i) | i <- [0 .. 9], s - i >= 0, s - i <= 9]
         in FI.logsumexp (Torch.stack (Torch.Dim 1) es) 1 False -- [B]
   in Torch.stack (Torch.Dim 1) [colFor s | s <- [0 .. 18]] -- [B,19]

-- ============================================================
--  BRIDGE: MeasU <-> GeomU (softmax decode to a Dist over digits)
-- ============================================================

instance MnistBridge MeasU GeomU where
  -- An image is the same pixel tensor in both universes (no representation gap,
  -- unlike binary's (Float,Float) -> tensor), so the encode is the identity.
  encImage :: Image MeasU -> Image GeomU
  encImage img = img

  decDigit :: Digit GeomU -> Dist (Digit MeasU)
  decDigit logits =
    let ps = head (Torch.asValue (F.softmax (F.Dim 1) logits) :: [[Float]])
     in FiniteSupp [(d, realToFrac (ps !! d)) | d <- [0 .. 9]]
