{-# LANGUAGE GADTs #-}

-- | Interpreters for 'LogVec'. The bind is folded in the LOG semiring: index
--   enumeration is host-side (cheap, over the finite support), the weight combination
--   is batched Torch ops (@Torch.add@ + @logsumexp@), so the result is differentiable.
--   The two-leaf fold of  @do { i <- a; j <- b; return (i + j) }@  IS the log-space
--   convolution -- the marginalization that used to be hand-coded as a log-space
--   convolution (@logConv@) now falls out of this fold for free.
module A_Categorical.Category.Monads.LogVecExpect
  ( logVecExpect,
    collectLeaves,
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

-- | Collect the leaf weight-tensors of an /applicative/ 'LogVec' chain -- each a @[B,k]@
--   tensor -- together with a reconstructor from a chosen index-combo (one index per leaf,
--   in order) to the final value. Assumes the chain is applicative: each leaf's structure
--   is independent of the earlier bound values (the final value may depend on all of them),
--   which holds for the universe-polymorphic formulas here. This lets a readout vectorize
--   the marginalization into a few batched tensor ops instead of an @O(prod k_i)@ scalar
--   fold, while computing exactly what 'logVecExpect' (the spec) would.
collectLeaves :: LogVec a -> ([Torch.Tensor], [Int] -> a)
collectLeaves (Pure x) = ([], const x)
collectLeaves (LogLeaf xs lw) = ([lw], \is -> xs !! head is)
collectLeaves (Bind m k) =
  let (lwsM, valsM) = collectLeaves m
      nM = length lwsM
      (lwsK, _) = collectLeaves (k (valsM (replicate nM 0)))
   in ( lwsM ++ lwsK,
        \is -> let (isM, isK) = splitAt nM is
                in snd (collectLeaves (k (valsM isM))) isK
      )

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
