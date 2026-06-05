{-# LANGUAGE GADTs #-}

-- | Interpreters for 'LogVec'. The bind is folded in the LOG semiring: index
--   enumeration is host-side (cheap, over the finite support), the weight combination
--   is batched Torch ops (@Torch.add@ + @logsumexp@), so the result is differentiable.
--   The two-leaf fold of  @do { i <- a; j <- b; return (i + j) }@  IS the log-space
--   convolution -- the marginalization that used to be hand-coded as a log-space
--   convolution (@logConv@) now falls out of this fold for free, vectorized by 'marginalize'.
module A_Categorical.Monads.LogVecExpect
  ( marginalize,
    mapLeafWeights,
    logVecLeafTensor,
  )
where

import A_Categorical.Monads.LogVec (LogVec (..))
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | Collect the leaf weight-tensors of an /applicative/ 'LogVec' chain -- each a @[B,k]@
--   tensor -- together with a reconstructor from a chosen index-combo (one index per leaf,
--   in order) to the final value. Assumes the chain is applicative: each leaf's structure
--   is independent of the earlier bound values (the final value may depend on all of them),
--   which holds for the monad-polymorphic formulas here. This lets 'marginalize' vectorize
--   the marginalization into a few batched tensor ops instead of an @O(prod k_i)@ scalar
--   fold, while computing exactly the log-semiring expectation (@logsumexp@ over the weighted
--   support). Internal to this module.
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

-- | The vectorized marginalization engine: collect the (independent) leaves of a 'LogVec' term,
--   build the joint log-weight @[B, k_0, ..., k_{n-1}]@ over all index-combos, and return the pair
--   @(logNum, logDen)@ (each a @[B]@ tensor) -- @logDen = logsumexp@ over ALL combos, @logNum =
--   logsumexp@ over the SAT combos. The caller supplies the SAT mask as a function of the combos'
--   VALUES (in flat row-major order): a @0/1@ tensor, either @[total]@ (batch-independent) or
--   @[B,total]@ (per-row), which broadcasts against the joint. This is the ONE fast engine behind
--   the @Bool@ quantifier ('B_Logical.Interpretations.TensorBool.logNumDen'); the per-row-mask
--   option keeps it ready for a per-row target predicate too. A few batched ops; no host-side fold.
marginalize :: LogVec a -> ([a] -> Torch.Tensor) -> (Torch.Tensor, Torch.Tensor)
marginalize prog satMask =
  let (lws, vals) = collectLeaves prog
      n = length lws
      ks = [Torch.shape lw !! 1 | lw <- lws] -- support sizes
      b = Torch.shape (head lws) !! 0 -- batch
      total = product ks
      reshapeFor i lw = Torch.reshape (b : [if j == i then ks !! j else 1 | j <- [0 .. n - 1]]) lw
      joint = foldr1 Torch.add [reshapeFor i lw | (i, lw) <- zip [0 ..] lws] -- broadcast each leaf over its own axis
      jointFlat = Torch.reshape [b, total] joint
      logDen = FI.logsumexp jointFlat 1 False -- [B]
      combos = sequence [[0 .. k - 1] | k <- ks]
      mask = satMask [vals c | c <- combos] -- [total] or [B,total], 1 = SAT
      logMask = (mask `Torch.sub` Torch.onesLike mask) `Torch.mul` Torch.asTensor (1.0e9 :: Float)
      logNum = FI.logsumexp (jointFlat `Torch.add` logMask) 1 False -- [B]
   in (logNum, logDen)

-- | The raw @[B,k]@ log-weight tensor of a leaf (for argmax-style decoding, e.g. the
--   digit-accuracy metric). Errors if not a 'LogLeaf'.
logVecLeafTensor :: LogVec a -> Torch.Tensor
logVecLeafTensor (LogLeaf _ lw) = lw
logVecLeafTensor _ = error "logVecLeafTensor: not a LogLeaf"

-- | Apply a tensor map to a leaf's @[B,k]@ weights, keeping its support (e.g. to gather/slice a
--   batched observation leaf along the batch dim 0 -- mini-batching the data without leaving the
--   monad). Errors on a non-leaf (a batched observation is always a single 'LogLeaf').
mapLeafWeights :: (Torch.Tensor -> Torch.Tensor) -> LogVec a -> LogVec a
mapLeafWeights f (LogLeaf xs lw) = LogLeaf xs (f lw)
mapLeafWeights _ _ = error "mapLeafWeights: expected a single LogLeaf"
