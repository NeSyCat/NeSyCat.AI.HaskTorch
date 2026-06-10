{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Logical interpretation: the @LogVec@ QUANTIFIER for the crisp @Bool@ truth algebra -- the
--   differentiable sibling of @Dist@'s quantifier in "B_Logical.Interpretations.Boolean".
--   The truth object is just @Bool@, SHARED with @Dist@ (the connectives live in Boolean's
--   universe-free @instance TwoMonBLat Bool@); this module adds only @A2MonBLat _ LogVec Bool@.
--   So @LogVec@ mirrors @Dist@ one-for-one, same truth algebra, only the monad differs:
--
--     @Dist Bool@  <->  @LogVec Bool@,   marginalize via 'logNumDen',   @eqNat = (==)@.
--
--   This is the probabilistic (DeepProbLog-style) reading: the truth values are crisp, the
--   marginalization (the law of total probability) is the 'LogVec' bind, and TRAINING STAYS IN
--   LOGITS. The ONE readout this module exposes is 'logNumDen' -- the log-space marginal
--   @(logNum, logDen)@ of a @LogVec Bool@ formula over its raw leaf logits (pure 'logsumexp', no
--   @exp@/clamp). Every consumer derives from it WHERE IT IS USED, not here: the negative-log
--   satisfaction @logDen - logNum@ inside the @lossKnow@ instance (the F inference layer), the
--   @[0,1]@ probability @exp(logNum - logDen)@ wherever a reading is wanted (the @decode@/@Dist@
--   bridge). So nothing on the training path ever forms a probability.
--
--   'bigWedge' (the @forall@ over the batch) takes the per-element @(logNum, logDen)@ marginal and
--   MEANS each over the batch (the product t-norm = @Dist@'s @bigWedge = product@) -- a real mean in
--   RAW log space (mean is linear, so @mean logDen - mean logNum =@ the mean NLL). The aggregate is
--   carried verbatim as a raw 'A_Categorical.Monads.LogVec.LogReduced' degree, NOT a normalized
--   Bernoulli, so no complement mass / @exp@ is ever formed on the training path; calibration to a
--   probability stays at the @decode@/@Dist@ bridge, never here. Reuses @Guard LogVec a = a@ and the crisp
--   @TwoMonBLat Bool@ from "Boolean".
module B_Logical.Interpretations.TensorBool
  ( logNumDen,
    module B_Logical.Signature.A2MonBLat,
  )
where

import A_Categorical.Monads.LogVec (LogVec (..))
import A_Categorical.Monads.LogVecExpect (collectLeaves, logConvolve, marginalize)
import B_Logical.Interpretations.Boolean () -- reuse: the universe-free @instance TwoMonBLat Bool@
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | In the @LogVec@ reading the guard IS the batched data itself: the vectorized predicate is
--   applied to the whole batch and reduced (polymorphic in the point type, mirroring
--   @Guard Dist a = [a]@).
type instance Guard LogVec a = a

------------------------------------------------------
-- A2MonBLat: the @LogVec@ quantifier interpretation for the crisp @Bool@ truth object --
-- polymorphic in the point type @a@ (apply the vectorized predicate to the batched guard,
-- read out, reduce over the batch). The connectives come from Boolean's @TwoMonBLat Bool@.
------------------------------------------------------

instance A2MonBLat LogVec Bool where
  -- forall = product t-norm over the batch = mean of the per-element negative-log satisfaction. We
  -- take the per-element @(logNum, logDen)@ marginal and MEAN each over the batch -- a real mean on
  -- the RAW log-masses. Mean is linear, so @mean logDen - mean logNum = mean (logDen - logNum) =@
  -- the mean NLL, with NO normalized Bernoulli and NO @exp@/complement on the training path. The
  -- aggregate is carried verbatim as a raw 'LogReduced' degree; calibration to a probability happens
  -- only at a readout (the @decode@/@Dist@ bridge), never on this path.
  bigWedge _ g phi =
    let (logNum, logDen) = logNumDen (phi g)
     in LogReduced (Torch.mean logNum) (Torch.mean logDen)
  bigVee _ _ _ = error "bigVee over LogVec Bool not yet supported in log space"
  bigOplus _ _ = error "bigOplus over LogVec Bool not yet supported"
  bigOtimes _ _ = error "bigOtimes over LogVec Bool not yet supported"

------------------------------------------------------
-- Readouts: one log-space marginalization, two views of it.
------------------------------------------------------

-- | The log-space marginalization of a 'LogVec Bool' formula, as a pair @(logNum, logDen)@
--   (each a @[B]@ tensor): @logDen = logsumexp@ over ALL outcome-combos, @logNum = logsumexp@
--   over the SAT-true ones.
--
--   Efficient marginalization REQUIRES structure -- marginalizing an arbitrary predicate is
--   @O(prod k_i)@ (the treewidth bound; no free lunch). So this exploits the ONE structure the
--   NeSy-arithmetic tasks have: when the predicate is an equality @observation == f(latents)@
--   with @f@ SEPARABLE (@f = base + sum_i c_i(x_i)@ for arbitrary per-leaf integer @c_i@), the
--   marginalization EMERGES from the bind as VARIABLE ELIMINATION / a log-space convolution
--   ('logNumDenConv') -- folding the leaves with no joint. This is NOT specialized to @(+)@: the
--   @c_i@ are discovered by probing the predicate, so sums, weighted sums (@10*d1+d2@), counts,
--   squares, Binary's iff (@c_i@ = identity) all take this path through the same code. Any
--   predicate that is NOT a separable-observation equality (a product, a @max@, an inequality, a
--   conjunction) fails the probe and falls back to the general full-joint 'marginalize' -- still
--   correct, just not scalable. (Efficiency for non-separable STRUCTURED predicates is what
--   knowledge compilation / arithmetic circuits, a la DeepProbLog, would add -- a separate engine.)
logNumDen :: LogVec Bool -> (Torch.Tensor, Torch.Tensor)
logNumDen (LogReduced logNum logDen) = (logNum, logDen) -- already marginalized: read the raw pair verbatim
logNumDen prog = case logNumDenConv prog of
  Just r -> r
  Nothing -> marginalize prog (\vs -> Torch.asTensor [if v then 1.0 else 0.0 :: Float | v <- vs])

-- | The convolution reading of a 'LogVec Bool' formula of the shape
--   @return (obs .= additiveFunctionOf latents)@: the observation is the LAST bound leaf, and the
--   predicate is an equality between the observation index and an additive combination of the
--   other (latent) leaves' indices. Returns @(logNum, logDen)@ folded by 'logConvolve' (no joint),
--   or @Nothing@ if the formula is not that pattern (verified by an additivity + indicator check
--   on the corner combo), so the caller falls back to 'marginalize'.
--
--   @logDen@ factorizes over the independent leaves (@sum_i logsumexp(leaf_i)@); @logNum =
--   logsumexp_v (sumDist[v] + obs[v])@ where @sumDist@ is the latent leaves convolved onto the
--   observation's value axis. Bit-identical to 'marginalize' (verified on single-digit).
logNumDenConv :: LogVec Bool -> Maybe (Torch.Tensor, Torch.Tensor)
logNumDenConv prog =
  let (lws, vals) = collectLeaves prog
      n = length lws
   in if n < 2
        then Nothing
        else
          let obsW = last lws
              latentWs = init lws
              nd = length latentWs
              ks = [Torch.shape w !! 1 | w <- latentWs]
              kObs = Torch.shape obsW !! 1
              maxSum = kObs - 1
              predicted dIdx = case [j | j <- [0 .. kObs - 1], vals (dIdx ++ [j])] of
                (j : _) -> Just j
                [] -> Nothing
              eVec i x = [if t == i then x else 0 | t <- [0 .. nd - 1]]
           in case predicted (replicate nd 0) of
                Nothing -> Nothing
                Just base ->
                  case sequence [sequence [subtract base <$> predicted (eVec i x) | x <- [0 .. ks !! i - 1]] | i <- [0 .. nd - 1]] of
                    Nothing -> Nothing
                    Just contribs ->
                      let maxCombo = [ks !! i - 1 | i <- [0 .. nd - 1]]
                          addOK = predicted maxCombo == Just (base + sum [contribs !! i !! (ks !! i - 1) | i <- [0 .. nd - 1]])
                          indOK = case predicted maxCombo of
                            Just pm -> vals (maxCombo ++ [pm]) && (kObs <= 1 || not (vals (maxCombo ++ [(pm + 1) `mod` kObs])))
                            Nothing -> False
                       in if not (addOK && indOK)
                            then Nothing
                            else
                              let sumDist = logConvolve maxSum base (zip contribs latentWs)
                                  logNum = FI.logsumexp (sumDist `Torch.add` obsW) 1 False
                                  logDen = foldr1 Torch.add [FI.logsumexp w 1 False | w <- lws]
                               in Just (logNum, logDen)
