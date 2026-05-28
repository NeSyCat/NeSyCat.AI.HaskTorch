{-# LANGUAGE GADTs #-}

-- | Interpreters for 'LogVec'. The bind is folded in the LOG semiring: index
--   enumeration is host-side (cheap, over the finite support), the weight combination
--   is batched Torch ops (@Torch.add@ + @logsumexp@), so the result is differentiable.
--   The two-leaf fold of  @do { i <- a; j <- b; return (i + j) }@  IS the log-space
--   convolution -- the marginalization that used to be hand-coded as a log-space
--   convolution (@logConv@) now falls out of this fold for free.
module A_Categorical.Category.Monads.LogVecExpect
  ( logVecExpect,
    logVecRunPure,
    logVecLeafTensor,
  )
where

import A_Categorical.Category.Monads.LogVec (LogVec (..))
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | Log-domain expectation: returns the @[B]@ tensor
--   @log Σ_x exp(logweight(x) + f x)@, folding the free monad. This is @distExpect@
--   over the log semiring @(Tensor[B], logsumexp, +)@ instead of @(Double, +, *)@.
--   Index enumeration (@xs@, @xs !! j@) is host-side; the @logsumexp@ over the
--   per-support contributions keeps autograd through the leaf weights @lw@.
logVecExpect :: LogVec a -> (a -> Torch.Tensor) -> Torch.Tensor
logVecExpect (Pure x) f = f x
logVecExpect (Bind m k) f = logVecExpect m (\x -> logVecExpect (k x) f)
logVecExpect (LogLeaf xs lw) f =
  FI.logsumexp
    (Torch.stack (Torch.Dim 1) [FI.select lw 1 j `Torch.add` f (xs !! j) | j <- [0 .. length xs - 1]])
    1
    False

-- | Collapse a deterministic 'LogVec' (only 'Pure'/'Bind', or a one-point 'LogLeaf')
--   to its value. Used where the program carries no genuine spread (e.g. Binary's
--   @classifierA = pure logits@). Errors on a multi-point 'LogLeaf'.
logVecRunPure :: LogVec a -> a
logVecRunPure (Pure x) = x
logVecRunPure (Bind m k) = logVecRunPure (k (logVecRunPure m))
logVecRunPure (LogLeaf [x] _) = x
logVecRunPure (LogLeaf _ _) = error "logVecRunPure: non-deterministic LogVec (multi-point LogLeaf)"

-- | The raw @[B,k]@ log-weight tensor of a leaf (for argmax-style decoding, e.g. the
--   digit-accuracy metric). Errors if not a 'LogLeaf'.
logVecLeafTensor :: LogVec a -> Torch.Tensor
logVecLeafTensor (LogLeaf _ lw) = lw
logVecLeafTensor _ = error "logVecLeafTensor: not a LogLeaf"
