{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}

-- | The LogTens monad: the log-space sibling of 'Dist'. A free monad
--   whose leaves carry a batched, log-weighted finite support -- i.e. the
--   function-space / free-module functor  @Tens a = (a -> R)@  realized over the LOG
--   semiring @(R, +, LogSumExp)@ with the weights kept as a 'Torch.Tensor' so autograd
--   survives. Its Kleisli BIND is the (log-space) convolution / law of total
--   probability: this is what lets the do-notation lift @plus = (+)@ to the
--   convolution automatically, exactly as the 'Dist' bind does for probabilities.
--
--   Finiteness is NOT intrinsic to the monad -- the construction @a |-> (a -> R)@ is
--   the representable functor for ANY @a@. It is only what the interpreter
--   ("A_Categorical.Monads.LogTensExpect") needs to MATERIALIZE the bind as a
--   dense tensor (a finite sum = the discrete convolution). The continuous case
--   (an integral = the general convolution) is the 'Giry' monad.
module A_Categorical.Monads.LogTens
  ( LogTens (..),
  )
where

import Control.Monad (ap)
import qualified Torch

-- | A finitely-supported, batched, log-weighted distribution as a free monad.
data LogTens a where
  Pure :: a -> LogTens a
  Bind :: LogTens x -> (x -> LogTens a) -> LogTens a
  LogLeaf :: [a] -> Torch.Tensor -> LogTens a
  -- | @LogLeaf xs lw@: support @xs@ (length @k@) with per-batch log-weights
  --   @lw :: [B,k]@ (column @j@ is the unnormalized log-weight of @xs !! j@).
  --   Autograd lives in @lw@; @xs@ is host data (the enumerable index set).
  -- | @LogReduced logNum logDen@: a @Bool@ sentence ALREADY marginalized to its raw log-masses --
  --   @logNum@ = log mass of the SAT outcomes, @logDen@ = log TOTAL mass (each a @[B]@ or scalar
  --   tensor). This is NOT a measure over @{True,False}@: @logDen@ counts mass off the enumerated
  --   support, so it is a @(numerator, denominator)@ SUMMARY, not a leaf. It lets the @forall@
  --   aggregation ("B_Logical.Interpretations.TensorBool".@bigWedge@) hold the batch-meaned
  --   satisfaction in RAW log space, so no normalized Bernoulli / complement mass (@log1mexp@, an
  --   @exp@) is ever formed on the training path -- calibration to a probability stays at a readout
  --   (the @decode@/@Dist@ bridge). Terminal: read out verbatim by
  --   @logNumDen@, never bound or @collectLeaves@d. (The one @Bool@-specific node in this otherwise
  --   polymorphic monad: a general @a@ reduces only to a full support vector, i.e. a 'LogLeaf'; the
  --   @(num, den)@ collapse is exactly the indicator/@Bool@ case.)
  LogReduced :: Torch.Tensor -> Torch.Tensor -> LogTens Bool

instance Functor LogTens where
  fmap :: (a -> b) -> LogTens a -> LogTens b
  fmap f m = Bind m (Pure . f)

instance Applicative LogTens where
  pure :: a -> LogTens a
  pure = Pure

  (<*>) :: LogTens (a -> b) -> LogTens a -> LogTens b
  (<*>) = ap

instance Monad LogTens where
  return :: a -> LogTens a
  return = pure

  (>>=) :: LogTens a -> (a -> LogTens b) -> LogTens b
  (>>=) = Bind
