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
import A_Categorical.Monads.LogVecExpect (marginalize)
import B_Logical.Interpretations.Boolean () -- reuse: the universe-free @instance TwoMonBLat Bool@
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import qualified Torch

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

-- | The log-space marginalization of a 'LogVec Bool' formula over its outcome-combos, as a
--   pair @(logNum, logDen)@ (each a @[B]@ tensor): @logDen = logsumexp@ over ALL combos,
--   @logNum = logsumexp@ over the SAT-true combos. Pure @logsumexp@ on the raw leaf logits --
--   no probability is formed (the convolution / law of total probability). Just the shared
--   vectorized 'marginalize' engine, with the SAT mask = the formula's own (batch-independent)
--   @Bool@ at each combo.
logNumDen :: LogVec Bool -> (Torch.Tensor, Torch.Tensor)
logNumDen prog = marginalize prog (\vs -> Torch.asTensor [if v then 1.0 else 0.0 :: Float | v <- vs])

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
