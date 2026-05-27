{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | Logical interpretation: fuzzy (product t-norm) logic on truth degrees in [0,1]
--   in the GeomU universe -- the LTN semantics. This is the probability-space sibling
--   of "B_Logical.Interpretations.Tensor" (which works on logits in \mathbb{R}):
--
--     * connectives:  neg = 1 - x,  otimes = product (the t-norm),  oplus / vee =
--       probabilistic sum (the t-conorm),  wedge = its De Morgan dual.
--     * quantifiers:  bigVee = pMean (\exists, with exponent @p = ParamsLogic@);
--       bigWedge = the PRODUCT t-norm aggregation (\forall), computed as a geometric
--       mean @exp(mean(log .))@ to avoid underflow -- the GeomU shadow of the MeasU
--       @bigWedge = product@ (so \forall reads the same in both universes). Paired with
--       @lossKnow = negLog@ (see "F_Inferential.InferenceInterpretation") this gives
--       @-log(geomean_i s_i) = mean_i(-log s_i)@, the categorical NLL -- whose steep
--       per-point gradient avoids the collapsed optimum a soft p-mean-error falls into.
--
--   The point type stays @Torch.Tensor@ (a batch), reusing the @Guard GeomU
--   Torch.Tensor@ instance from "B_Logical.Interpretations.Tensor".
module B_Logical.Interpretations.TensorProb
  ( OmegaP (..),
    module B_Logical.Signature.TwoMonBLat,
    module B_Logical.Signature.A2MonBLat,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.Tensor () -- reuse: type instance Guard GeomU Torch.Tensor
import B_Logical.Library.PMean (pMean)
import B_Logical.Library.Stable (clampNotZero)
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import Data.Functor.Identity (Identity (..), runIdentity)
import qualified Torch

-- | Omega_P := the fuzzy truth object I(tau) = [0,1] (a tensor of degrees). A newtype
--   so its instances do not overlap the logit @Omega = Torch.Tensor@ of "Tensor".
newtype OmegaP = OmegaP {unOmegaP :: Torch.Tensor}

------------------------------------------------------
-- TwoMonBLat: connectives on Omega_P ([0,1]-valued, product fuzzy logic)
------------------------------------------------------

instance TwoMonBLat GeomU OmegaP where
  type ParamsLogic OmegaP = Float -- the LTN aggregator exponent p
  top = OmegaP (Torch.asTensor [1.0 :: Float])
  bot = OmegaP (Torch.asTensor [0.0 :: Float])
  neg (OmegaP x) = OmegaP (Torch.onesLike x `Torch.sub` x)
  -- disjunction: the probabilistic sum (t-conorm) a + b - a*b; serves as the join.
  vee _ (OmegaP a) (OmegaP b) = OmegaP ((a `Torch.add` b) `Torch.sub` (a `Torch.mul` b))
  -- the two monoids: oplus = t-conorm (unit 0), otimes = product t-norm (unit 1).
  oplus (OmegaP a) (OmegaP b) = OmegaP ((a `Torch.add` b) `Torch.sub` (a `Torch.mul` b))
  o0 = OmegaP (Torch.asTensor [0.0 :: Float])
  otimes (OmegaP a) (OmegaP b) = OmegaP (a `Torch.mul` b)
  o1 = OmegaP (Torch.asTensor [1.0 :: Float])
  vdash (OmegaP a) (OmegaP b) = Torch.asValue a <= (Torch.asValue b :: Float)

------------------------------------------------------
-- A2MonBLat: quantifiers via the LTN generalized means
------------------------------------------------------

instance A2MonBLat Torch.Tensor GeomU OmegaP where
  -- bigVee = exists = pMean(p)
  bigVee p guard phi =
    let OmegaP result = runIdentity (phi guard)
     in Identity (OmegaP (pMean p 0 result))
  -- bigWedge = forall = the PRODUCT t-norm aggregation, as a geometric mean (the
  -- exponent p is unused: the universal product is parameter-free). Matches MeasU's
  -- bigWedge = product; with lossKnow = negLog it yields the categorical NLL.
  bigWedge _p guard phi =
    let OmegaP result = runIdentity (phi guard)
     in Identity (OmegaP (geoMean 0 result))
  bigOplus _ _ = error "bigOplus over OmegaP not yet supported"
  bigOtimes _ _ = error "bigOtimes over OmegaP not yet supported"

-- | Geometric mean over dimension @d@: @exp(mean_d(log x))@ (the product t-norm
--   aggregation in stable log form). Inputs clamped away from 0 so @log@ stays finite.
geoMean :: Int -> Torch.Tensor -> Torch.Tensor
geoMean d x =
  let n = Torch.shape x !! d
      lx = Torch.log (clampNotZero 1e-4 x)
   in Torch.exp (Torch.sumDim (Torch.Dim d) Torch.RemoveDim Torch.Float lx `Torch.div` Torch.asTensor (fromIntegral n :: Float))
