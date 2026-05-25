{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Interpretation I_gamma for Binary Classification: the assignment of the
--   signature symbols to objects/morphisms of the domain category.
--
--     Sorts       : I(Point), I(Omega)        per universe (MeasU, GeomU)
--     Theta       : I(Theta) = ParamsMLP      (the MLP weight space, an actor object)
--     labelA      : the circle-in-square ground truth
--     classifierA : the MLP morphism hThetaReal -- the network *is* this semantics
--     bridge      : encPoint / decOmega between MeasU and GeomU
module Examples.Binary.C_Domain.Interpretation
  ( module Examples.Binary.C_Domain.Signature,
    module Lib.C_Domain.Models.MLP,
  )
where

import Lib.A_Categorical.CategoricalInterpretation (GeomU, MeasU)
import Lib.A_Categorical.Category.Monads.Dist (Dist (..))
import qualified Lib.B_Logical.Interpretations.Boolean as BoolLogic
import Lib.B_Logical.Interpretations.Tensor hiding (Omega)
import qualified Lib.B_Logical.Interpretations.Tensor as TensLogic
import Lib.C_Domain.Models.MLP (ParamsMLP, binarySpecReal, hThetaReal)
import Examples.Binary.C_Domain.Signature (BinaryBridge (..), BinaryKlRel (..), BinaryRel (..), BinarySorts (..))
import Data.Functor.Identity (Identity (..))
import qualified Torch
import qualified Torch.Functional.Internal as F

-- ============================================================
--  Sort assignments: I(Point), I(Omega)
-- ============================================================

instance BinarySorts MeasU where
  type Point MeasU = (Float, Float)   -- R^2 as a Cartesian product
  type Omega MeasU = BoolLogic.Omega  -- = Bool

instance BinarySorts GeomU where
  type Point GeomU = Torch.Tensor     -- shape: [2], dtype: Float
  type Omega GeomU = TensLogic.Omega  -- = Torch.Tensor

-- ============================================================
--  MeasU: plain relation symbols (BinaryRel)
-- ============================================================

instance BinaryRel MeasU where
  labelA :: Point MeasU -> Omega MeasU
  labelA (x1, x2) =
    let dx = x1 - 0.5
        dy = x2 - 0.5
     in dx * dx + dy * dy < 0.09

-- ============================================================
--  MeasU: Kleisli relation symbols (BinaryKlRel)
-- ============================================================

instance BinaryKlRel MeasU where
  type Theta MeasU = ParamsMLP
  classifierA :: ParamsMLP -> Point MeasU -> Dist (Omega MeasU)
  classifierA paramMLP pt =
    let ptTens = encPoint @MeasU @GeomU pt
        logits = hThetaReal paramMLP (Torch.reshape [1, 2] ptTens)
     in decOmega @MeasU @GeomU logits

-- ============================================================
--  GeomU: plain relation symbols (BinaryRel)
-- ============================================================

instance BinaryRel GeomU where
  -- | Label in GeomU: returns R logits (True = +logitScale, False = -logitScale).
  labelA :: Point GeomU -> Omega GeomU
  labelA pt =
    let center = F.mulScalar (Torch.onesLike pt) (0.5 :: Float)
        diff = pt `Torch.sub` center
        dist2 = Torch.sumDim (Torch.Dim (-1)) Torch.KeepDim Torch.Float (diff * diff)
        radiusSq = F.mulScalar (Torch.onesLike dist2) (0.09 :: Float)
        isInside = Torch.lt dist2 radiusSq
        boolFloat = Torch.toType Torch.Float isInside
        scale = F.mulScalar (Torch.onesLike boolFloat) logitScale
     in boolFloat `Torch.mul` (scale `Torch.add` scale) `Torch.sub` scale

logitScale :: Float
logitScale = 10.0

-- ============================================================
--  GeomU: Kleisli relation symbols (BinaryKlRel)
-- ============================================================

instance BinaryKlRel GeomU where
  type Theta GeomU = ParamsMLP
  classifierA :: ParamsMLP -> Point GeomU -> Identity (Omega GeomU)
  classifierA paramMLP ptTensor =
    Identity (hThetaReal paramMLP ptTensor)

-- ============================================================
--  BRIDGE: MeasU <-> GeomU (with Dist monad for decoding)
-- ============================================================

instance BinaryBridge MeasU GeomU where
  encPoint :: Point MeasU -> Point GeomU
  encPoint (x1, x2) =
    Torch.toDevice (Torch.Device Torch.CPU 0) (Torch.asTensor [x1, x2])

  decOmega :: Omega GeomU -> Dist (Omega MeasU)
  decOmega probs =
    let val = Torch.asValue (Torch.sigmoid probs) :: [[Float]]
        p = realToFrac (head (head val)) :: Double
     in FiniteSupp [(True, p), (False, 1.0 - p)]
