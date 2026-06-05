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
--     @Dist Bool@  <->  @LogVec Bool@,   @distPTrue@  <->  'logVecPTrue',   @eqNat = (==)@.
--
--   This is the probabilistic (DeepProbLog-style) reading: the truth values are crisp, the
--   marginalization (the law of total probability) is the 'LogVec' bind, and TRAINING STAYS IN
--   LOGITS. Both readouts come from the same log-space @(logNum, logDen)@ over the raw leaf
--   logits ('logNumDen'):
--
--     * 'logVecNLL' @= logDen - logNum@  -- the negative-log satisfaction, the LOSS readout.
--       Pure 'logsumexp' arithmetic on logits: no @exp@-to-probability, no clamp, full gradient.
--     * 'logVecPTrue' @= exp(logNum - logDen)@  -- the [0,1] probability, the READING readout
--       (the @LogVec@ twin of @distPTrue@); never on the training path.
--
--   'bigWedge' (the @forall@ over the batch) aggregates the per-element 'logVecNLL' in LOG space
--   (the mean = the product t-norm = @Dist@'s @bigWedge = product@) and returns the aggregate as
--   the sentence's truth: a Bernoulli leaf built directly in log space ('nllLeaf'), so the loss
--   reads it back exactly with no probability ever materialized. Reuses @Guard LogVec a = a@ and
--   the crisp @TwoMonBLat Bool@ from "Boolean".
module B_Logical.Interpretations.TensorBool
  ( logVecNLL,
    logVecPTrue,
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
  -- forall = product t-norm over the batch = mean of the per-element negative-log satisfaction,
  -- all in LOG space (no exp-to-probability, no clamp). The aggregate is returned as the
  -- sentence's truth Bernoulli, built in log space, so 'logVecNLL' reads it back exactly.
  bigWedge _ g phi = nllLeaf (Torch.mean (logVecNLL (phi g)))
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

-- | Negative-log satisfaction @-log P(true) = logDen - logNum@ (a @[B]@ tensor) -- the LOSS
--   readout, pure log space (a difference of @logsumexp@s = the log-domain cross-entropy / the
--   log-space convolution). No @exp@, no clamp: confidently-wrong outcomes keep full gradient.
logVecNLL :: LogVec Bool -> Torch.Tensor
logVecNLL m = let (logNum, logDen) = logNumDen m in logDen `Torch.sub` logNum

-- | Satisfaction probability @P(true) = exp(logNum - logDen)@ (a @[B]@ tensor) -- the READING
--   readout (the @LogVec@ twin of @distPTrue@). Never used on the training path.
logVecPTrue :: LogVec Bool -> Torch.Tensor
logVecPTrue m = let (logNum, logDen) = logNumDen m in Torch.exp (logNum `Torch.sub` logDen)

-- | Encode a negative-log-satisfaction scalar @s@ as the sentence's truth Bernoulli, in log
--   space: @LogLeaf [True,False] [-s, log(1 - e^{-s})]@. Normalized by construction, so
--   @logVecNLL (nllLeaf s) = s@ exactly (and @logVecPTrue (nllLeaf s) = e^{-s}@), with no
--   @exp@ of the aggregate and no clamp.
nllLeaf :: Torch.Tensor -> LogVec Bool
nllLeaf s =
  let ns = negate s
   in LogLeaf [True, False] (Torch.reshape [1, 2] (Torch.stack (Torch.Dim 0) [ns, log1mexp ns]))

-- | @log(1 - exp x)@ for @x <= 0@ (the log-complement of a log-probability).
log1mexp :: Torch.Tensor -> Torch.Tensor
log1mexp x = Torch.log (Torch.onesLike x `Torch.sub` Torch.exp x)
