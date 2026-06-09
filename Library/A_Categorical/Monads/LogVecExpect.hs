{-# LANGUAGE GADTs #-}

-- | Interpreters for 'LogVec'. The bind is folded in the LOG semiring: index
--   enumeration is host-side (cheap, over the finite support), the weight combination
--   is batched Torch ops, so the result is differentiable.
--
--   The marginalization genuinely EMERGES from the bind as a LOG-SPACE CONVOLUTION
--   ('logConvolve'): the do-block @do { i <- a; j <- b; return (i + j) }@ folds the leaves
--   one at a time, merging equal partial sums into a @[B, maxSum+1]@ tensor (variable
--   elimination), so the @O(prod k_i)@ joint NEVER materializes -- only @O(n * k * maxSum)@
--   work. 'marginalize' (the old full-joint reduction) is kept as a correctness oracle /
--   fallback for non-additive predicates.
module A_Categorical.Monads.LogVecExpect
  ( collectLeaves,
    logConvolve,
    marginalize,
    mapLeafWeights,
    logVecLeafTensor,
  )
where

import A_Categorical.Monads.LogVec (LogVec (..))
import Data.List (foldl')
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | Collect the leaf weight-tensors of an /applicative/ 'LogVec' chain -- each a @[B,k]@
--   tensor -- together with a reconstructor from a chosen index-combo (one index per leaf,
--   in order) to the final value. Assumes the chain is applicative: each leaf's structure
--   is independent of the earlier bound values (the final value may depend on all of them),
--   which holds for the monad-polymorphic formulas here.
collectLeaves :: LogVec a -> ([Torch.Tensor], [Int] -> a)
collectLeaves (Pure x) = ([], const x)
collectLeaves (LogLeaf xs lw) = ([lw], \is -> xs !! head is)
collectLeaves (LogReduced _ _) =
  error "collectLeaves: LogReduced is pre-marginalized -- read it via logNumDen, do not collect"
collectLeaves (Bind m k) =
  let (lwsM, valsM) = collectLeaves m
      nM = length lwsM
      (lwsK, _) = collectLeaves (k (valsM (replicate nM 0)))
   in ( lwsM ++ lwsK,
        \is -> let (isM, isK) = splitAt nM is
                in snd (collectLeaves (k (valsM isM))) isK
      )

-- | Per-bin log-sum-exp scatter -- the one delicate op of the convolution.
--   Given per-pair contributions @c :: [B,P]@ and a host bin-index vector @idx :: [P]@
--   (each in @[0, nbins)@), returns @out :: [B, nbins]@ with
--   @out[b,j] = log sum_{p : idx[p]==j} exp(c[b,p])@. Numerically stable via a per-row max
--   shift (the shift's gradient cancels analytically, so no detach is needed); empty bins get
--   @log(eps)+m@ (effectively @-inf@, zero mass) with a clean finite gradient.
logScatter :: Int -> [Int] -> Torch.Tensor -> Torch.Tensor
logScatter nbins idx c =
  let b = head (Torch.shape c)
      p = Torch.shape c !! 1
      m = FI.amax c 1 True -- [B,1] per-row max
      e = Torch.exp (c `Torch.sub` m) -- [B,P] in (0,1]
      idxBP = Torch.expand (Torch.reshape [1, p] (Torch.asTensor (idx :: [Int]))) False [b, p]
      acc = FI.scatterAdd (Torch.zeros' [b, nbins]) 1 idxBP e -- [B,nbins]
      eps = Torch.asTensor (1.0e-30 :: Float)
   in Torch.log (acc `Torch.add` eps) `Torch.add` m

-- | The log-space CONVOLUTION engine (variable elimination): fold integer-valued leaves into a
--   dense @[B, maxSum+1]@ log-marginal over their sum, one leaf at a time. Each leaf is given as
--   @(contributionValues, weights)@ -- @contributionValues !! x@ is the integer this leaf's index
--   @x@ adds to the running sum, @weights@ its @[B,k]@ log-weights. @base@ is the sum at the
--   all-zero combo (the dense starts as a log-delta at @base@). Equal partial sums merge via
--   'logScatter', so the peak intermediate is @[B, (maxSum+1)*k]@, never the @O(prod k_i)@ joint.
logConvolve :: Int -> Int -> [([Int], Torch.Tensor)] -> Torch.Tensor
logConvolve maxSum base leaves =
  let v = maxSum + 1
      b = head (Torch.shape (snd (head leaves)))
      negBig = -1.0e9 :: Float
      dense0 =
        Torch.expand
          (Torch.asTensor ([[if j == base then 0.0 else negBig | j <- [0 .. maxSum]]] :: [[Float]]))
          False
          [b, v]
      step dense (cvs, lw) =
        let k = length cvs
            cMat = Torch.reshape [b, v * k] (Torch.reshape [b, v, 1] dense `Torch.add` Torch.reshape [b, 1, k] lw)
            idx = [max 0 (min maxSum (a + cvs !! x)) | a <- [0 .. maxSum], x <- [0 .. k - 1]]
         in logScatter v idx cMat
   in foldl' step dense0 leaves

-- | The vectorized full-joint marginalization -- KEPT as the correctness oracle and the fallback
--   for predicates that are not an equality against an additive function of the leaves (the
--   convolution handles those; see 'B_Logical.Interpretations.TensorBool.logNumDen'). Builds the
--   joint @[B, k_0, ..., k_{n-1}]@ and returns @(logNum, logDen)@ via @logsumexp@.
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
