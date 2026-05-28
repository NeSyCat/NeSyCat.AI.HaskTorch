{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Logical interpretation: fuzzy (product t-norm) logic on truth degrees in [0,1] in
--   the GeomU universe -- the probability-space sibling of "B_Logical.Interpretations.Tensor"
--   (which works on logits in \mathbb{R}):
--
--     * connectives:  neg = 1 - x,  otimes = product (the t-norm),  oplus / vee =
--       probabilistic sum (the t-conorm),  wedge = its De Morgan dual. Parameter-free
--       (@ParamsLogic OmegaP = ()@), so the universal carries no smoothing knob.
--     * quantifiers:  bigWedge = the PRODUCT t-norm aggregation (\forall), as a geometric
--       mean @exp(mean(log .))@ (stable; the GeomU shadow of the MeasU @bigWedge = product@,
--       so \forall reads the same in both universes); bigVee its De Morgan dual (\exists).
--       Paired with @lossKnow = negLog@ this gives @-log(geomean s) = mean(-log s)@, the NLL.
--
--   The 'A2MonBLat' instance is POLYMORPHIC in the point type @a@ (any batched data shape:
--   single tensors, tuples, ...), exactly like Boolean's @A2MonBLat a MeasU Bool@: in GeomU
--   the (vectorized) predicate is applied to the batched guard and reduced over the batch,
--   so no per-example/per-shape instance is ever needed. Reuses @Guard GeomU a = a@ from "Tensor".
module B_Logical.Interpretations.TensorProb
  ( OmegaP (..),
    module B_Logical.Signature.TwoMonBLat,
    module B_Logical.Signature.A2MonBLat,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import A_Categorical.Category.Monads.LogVecExpect (collectLeaves)
import B_Logical.Interpretations.Tensor () -- reuse: type instance Guard GeomU a = a
import B_Logical.Library.Stable (clampNotZero)
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | Omega_P := the fuzzy truth object I(tau) = [0,1] (a tensor of degrees). A newtype
--   so its instances do not overlap the logit @Omega = Torch.Tensor@ of "Tensor".
newtype OmegaP = OmegaP {unOmegaP :: Torch.Tensor}

------------------------------------------------------
-- TwoMonBLat: connectives on Omega_P ([0,1]-valued, product fuzzy logic; parameter-free)
------------------------------------------------------

instance TwoMonBLat GeomU OmegaP where
  type ParamsLogic OmegaP = () -- the product logic carries no smoothing parameter
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
-- A2MonBLat: the GeomU quantifier interpretation -- polymorphic in the point type @a@
-- (apply the vectorized predicate to the batched guard, then reduce over the batch).
------------------------------------------------------

instance A2MonBLat a GeomU OmegaP where
  -- forall = product t-norm over the batch (geometric mean). Matches MeasU's bigWedge=product.
  bigWedge _ g phi =
    let OmegaP r = logVecReadoutP (phi g)
     in Pure (OmegaP (geoMean 0 r))
  -- exists = the De Morgan dual (parameter-free); not exercised by current examples.
  bigVee _ g phi =
    let OmegaP r = logVecReadoutP (phi g)
        agg = geoMean 0 (Torch.onesLike r `Torch.sub` r)
     in Pure (OmegaP (Torch.onesLike agg `Torch.sub` agg))
  bigOplus _ _ = error "bigOplus over OmegaP not yet supported"
  bigOtimes _ _ = error "bigOtimes over OmegaP not yet supported"

-- | Geometric mean over dimension @d@: @exp(mean_d(log x))@ (the product t-norm
--   aggregation in stable log form). Inputs clamped away from 0 so @log@ stays finite.
geoMean :: Int -> Torch.Tensor -> Torch.Tensor
geoMean d x =
  let n = Torch.shape x !! d
      lx = Torch.log (clampNotZero 1e-4 x)
   in Torch.exp (Torch.sumDim (Torch.Dim d) Torch.RemoveDim Torch.Float lx `Torch.div` Torch.asTensor (fromIntegral n :: Float))

-- | Marginalize a 'LogVec'-valued GeomU formula to its satisfaction degree in [0,1]
--   (a @[B]@ tensor). The convolution is performed by the 'LogVec' bind inside
--   'logVecExpect'; this reads out @P(formula true)@ as the softmax-normalized
--   log-mass on the formula's true outcomes:
--     @P = exp( logsumexp_x(logw_x + log truth_x)  -  logsumexp_x(logw_x) )@.
--   For the MNIST axiom (truth_x = 1[d1+d2 = observed]) this is exactly the old
--   @probEq (oneHot n) (logConv lx ly)@ -- same forward and backward graph -- so the
--   hand-coded convolution is replaced with no change in behaviour.
logVecReadoutP :: LogVec OmegaP -> OmegaP
logVecReadoutP prog =
  let (lws, vals) = collectLeaves prog                          -- the chain's independent leaves [B,k_i]
      n = length lws
      ks = [Torch.shape lw !! 1 | lw <- lws]                    -- support sizes
      b = Torch.shape (head lws) !! 0                           -- batch
      total = product ks
      -- joint log-weight [B, k_0, ..., k_{n-1}] by broadcasting each leaf over its own axis
      reshapeFor i lw = Torch.reshape (b : [if j == i then ks !! j else 1 | j <- [0 .. n - 1]]) lw
      joint = foldr1 Torch.add [reshapeFor i lw | (i, lw) <- zip [0 ..] lws]
      jointFlat = Torch.reshape [b, total] joint
      logDen = FI.logsumexp jointFlat 1 False                   -- log Sum exp(logweights)   [B]
      -- truth mask over the index-combos (batch-independent 0/1), as a log-domain offset
      combos = sequence [[0 .. k - 1] | k <- ks]
      mask = Torch.reshape [total] (Torch.stack (Torch.Dim 0) [unOmegaP (vals c) | c <- combos])
      logMask = (mask `Torch.sub` Torch.onesLike mask) `Torch.mul` Torch.asTensor (1.0e9 :: Float)
      logNum = FI.logsumexp (jointFlat `Torch.add` Torch.reshape [1, total] logMask) 1 False
   in OmegaP (Torch.exp (logNum `Torch.sub` logDen))
