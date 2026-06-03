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
--   This is the probabilistic (DeepProbLog-style) reading: the truth values are crisp, the
--   marginalization (the law of total probability) is the 'LogVec' bind, and TRAINING STAYS IN
--   LOGITS. Both readouts come from the same log-space @(logNum, logDen)@ over the raw leaf
--   logits ('logNumDen'):
--
--     * 'logVecNLL' @= logDen - logNum@  -- the negative-log satisfaction, the LOSS readout.
--       Pure 'logsumexp' arithmetic on logits: no @exp@-to-probability, no clamp, full gradient.
--     * 'logVecPTrue' @= exp(logNum - logDen)@  -- the [0,1] probability, the READING readout
--       (the GeomU twin of @distPTrue@); never on the training path.
--
--   'bigWedge' (the @forall@ over the batch) aggregates the per-element 'logVecNLL' in LOG space
--   (the mean = the product t-norm = MeasU's @bigWedge = product@) and returns the aggregate as
--   the sentence's truth: a Bernoulli leaf built directly in log space ('nllLeaf'), so the loss
--   reads it back exactly with no probability ever materialized. Reuses @Guard GeomU a = a@ and
--   the crisp @TwoMonBLat Bool@ from "Boolean".
module B_Logical.Interpretations.TensorBool
  ( logVecNLL,
    logVecPTrue,
    module B_Logical.Signature.A2MonBLat,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import A_Categorical.Category.Monads.LogVec (LogVec (..))
import A_Categorical.Category.Monads.LogVecExpect (collectLeaves)
import B_Logical.Interpretations.Boolean () -- reuse: the universe-free @instance TwoMonBLat Bool@
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | In GeomU the guard IS the batched data itself: the vectorized predicate is applied to the
--   whole batch and reduced (polymorphic in the point type, mirroring @Guard MeasU a = [a]@).
type instance Guard GeomU a = a

------------------------------------------------------
-- A2MonBLat: the GeomU quantifier interpretation for the crisp @Bool@ truth object --
-- polymorphic in the point type @a@ (apply the vectorized predicate to the batched guard,
-- read out, reduce over the batch). The connectives come from Boolean's @TwoMonBLat Bool@.
------------------------------------------------------

instance A2MonBLat a GeomU Bool where
  -- forall = product t-norm over the batch = mean of the per-element negative-log satisfaction,
  -- all in LOG space (no exp-to-probability, no clamp). The aggregate is returned as the
  -- sentence's truth Bernoulli, built in log space, so 'logVecNLL' reads it back exactly.
  bigWedge _ g phi = nllLeaf (Torch.mean (logVecNLL (phi g)))
  bigVee _ _ _ = error "bigVee over GeomU Bool not yet supported in log space"
  bigOplus _ _ = error "bigOplus over GeomU Bool not yet supported"
  bigOtimes _ _ = error "bigOtimes over GeomU Bool not yet supported"

------------------------------------------------------
-- Readouts: one log-space marginalization, two views of it.
------------------------------------------------------

-- | The log-space marginalization of a 'LogVec Bool' formula over its outcome-combos, as a
--   pair @(logNum, logDen)@ (each a @[B]@ tensor): @logDen = logsumexp@ over ALL combos,
--   @logNum = logsumexp@ over the SAT-true combos. Pure @logsumexp@ on the raw leaf logits --
--   no probability is formed. This is the convolution / law of total probability.
logNumDen :: LogVec Bool -> (Torch.Tensor, Torch.Tensor)
logNumDen prog =
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
   in (logNum, logDen)

-- | Negative-log satisfaction @-log P(true) = logDen - logNum@ (a @[B]@ tensor) -- the LOSS
--   readout, pure log space (a difference of @logsumexp@s = the log-domain cross-entropy / the
--   log-space convolution). No @exp@, no clamp: confidently-wrong outcomes keep full gradient.
logVecNLL :: LogVec Bool -> Torch.Tensor
logVecNLL m = let (logNum, logDen) = logNumDen m in logDen `Torch.sub` logNum

-- | Satisfaction probability @P(true) = exp(logNum - logDen)@ (a @[B]@ tensor) -- the READING
--   readout (the GeomU twin of @distPTrue@). Never used on the training path.
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
