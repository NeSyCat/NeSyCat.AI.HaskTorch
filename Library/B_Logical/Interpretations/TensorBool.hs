{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Logical interpretation: the GeomU QUANTIFIER for the crisp @Bool@ truth algebra -- the
--   differentiable sibling of MeasU's quantifier in "B_Logical.Interpretations.Boolean".
--   The truth object is just @Bool@, SHARED with MeasU (the connectives live in Boolean's
--   universe-free @instance TwoMonBLat Bool@); this module adds only @A2MonBLat _ GeomU Bool@.
--   So GeomU mirrors MeasU one-for-one, same truth algebra, only the monad differs:
--
--     @Dist Bool@  <->  @LogVec Bool@,   @distPTrue@  <->  'logVecPTrue',   @eqNat = (==)@.
--
--   This is the probabilistic (DeepProbLog-style) reading: the truth values are crisp, and
--   the marginalization -- the law of total probability -- is supplied by the 'LogVec' bind.
--   'bigWedge' reads each batch element's @P(true)@ out via 'logVecPTrue', takes the product
--   t-norm ('geoMean', matching MeasU's @bigWedge = product@), and returns the aggregate as a
--   Bernoulli leaf (a one-leaf @LogVec Bool@ carrying the degree in its weights). Reuses
--   @Guard GeomU a = a@ from "Tensor" and the crisp @TwoMonBLat Bool@ from "Boolean".
module B_Logical.Interpretations.TensorBool
  ( logVecPTrue,
    bernoulli,
    module B_Logical.Signature.A2MonBLat,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import A_Categorical.Category.Monads.LogVecExpect (collectLeaves)
import B_Logical.Interpretations.Boolean () -- reuse: the universe-free @instance TwoMonBLat Bool@
import B_Logical.Interpretations.Tensor () -- reuse: type instance Guard GeomU a = a
import B_Logical.Library.Stable (clampNotZero)
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import qualified Torch
import qualified Torch.Functional.Internal as FI

------------------------------------------------------
-- A2MonBLat: the GeomU quantifier interpretation for the crisp @Bool@ truth object --
-- polymorphic in the point type @a@ (apply the vectorized predicate to the batched guard,
-- read out, reduce over the batch). The connectives come from Boolean's @TwoMonBLat Bool@.
------------------------------------------------------

instance A2MonBLat a GeomU Bool where
  -- forall = product t-norm over the batch (geometric mean). Matches MeasU's bigWedge=product.
  bigWedge _ g phi = bernoulli (geoMean 0 (logVecPTrue (phi g)))
  -- exists = the De Morgan dual (parameter-free); not exercised by current examples.
  bigVee _ g phi =
    let r = logVecPTrue (phi g)
        agg = geoMean 0 (Torch.onesLike r `Torch.sub` r)
     in bernoulli (Torch.onesLike agg `Torch.sub` agg)
  bigOplus _ _ = error "bigOplus over GeomU Bool not yet supported"
  bigOtimes _ _ = error "bigOtimes over GeomU Bool not yet supported"

-- | A (certain-or-uncertain) Bernoulli over @\{True, False\}@ carrying the satisfaction
--   degree @p@ in its log-weights: @LogLeaf [True, False] [log p, log (1-p)]@ (a @[1,2]@
--   tensor). The GeomU analogue of @FiniteSupp [(True, p), (False, 1-p)]@; 'logVecPTrue'
--   of it is @p@ again, so the degree survives the round-trip exactly.
bernoulli :: Torch.Tensor -> LogVec Bool
bernoulli p =
  LogLeaf
    [True, False]
    (Torch.reshape [1, 2] (Torch.stack (Torch.Dim 0) [Torch.log p, Torch.log (Torch.onesLike p `Torch.sub` p)]))

-- | Geometric mean over dimension @d@: @exp(mean_d(log x))@ (the product t-norm
--   aggregation in stable log form). Inputs clamped away from 0 so @log@ stays finite.
geoMean :: Int -> Torch.Tensor -> Torch.Tensor
geoMean d x =
  let n = Torch.shape x !! d
      lx = Torch.log (clampNotZero 1e-4 x)
   in Torch.exp (Torch.sumDim (Torch.Dim d) Torch.RemoveDim Torch.Float lx `Torch.div` Torch.asTensor (fromIntegral n :: Float))

-- | Marginalize a 'LogVec'-valued GeomU formula to its satisfaction degree in [0,1]
--   (a @[B]@ tensor) -- the GeomU twin of MeasU's @distPTrue m = distExpect m (\\b -> if b
--   then 1 else 0)@. The convolution is performed by the 'LogVec' bind; this reads out
--   @P(formula true)@ as the softmax-normalized log-mass on the formula's true outcomes:
--     @P = exp( logsumexp_x(logw_x + log truth_x)  -  logsumexp_x(logw_x) )@,
--   where @truth_x = 1@[the @Bool@ at index-combo @x@]. For the MNIST axiom
--   (@truth_x = 1[d1+d2 = observed]@) this is exactly the old softmax-of-logConv -- same
--   forward and backward graph.
logVecPTrue :: LogVec Bool -> Torch.Tensor
logVecPTrue prog =
  let (lws, vals) = collectLeaves prog -- the chain's independent leaves [B,k_i]
      n = length lws
      ks = [Torch.shape lw !! 1 | lw <- lws] -- support sizes
      b = Torch.shape (head lws) !! 0 -- batch
      total = product ks
      -- joint log-weight [B, k_0, ..., k_{n-1}] by broadcasting each leaf over its own axis
      reshapeFor i lw = Torch.reshape (b : [if j == i then ks !! j else 1 | j <- [0 .. n - 1]]) lw
      joint = foldr1 Torch.add [reshapeFor i lw | (i, lw) <- zip [0 ..] lws]
      jointFlat = Torch.reshape [b, total] joint
      logDen = FI.logsumexp jointFlat 1 False -- log Sum exp(logweights)   [B]
      -- truth mask over the index-combos (batch-independent 0/1), as a log-domain offset
      combos = sequence [[0 .. k - 1] | k <- ks]
      truth c = Torch.asTensor [if vals c then 1.0 else 0.0 :: Float]
      mask = Torch.reshape [total] (Torch.stack (Torch.Dim 0) [truth c | c <- combos])
      logMask = (mask `Torch.sub` Torch.onesLike mask) `Torch.mul` Torch.asTensor (1.0e9 :: Float)
      logNum = FI.logsumexp (jointFlat `Torch.add` Torch.reshape [1, total] logMask) 1 False
   in Torch.exp (logNum `Torch.sub` logDen)
