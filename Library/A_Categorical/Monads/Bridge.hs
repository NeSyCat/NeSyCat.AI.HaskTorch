{-# LANGUAGE GADTs #-}

-- | The two bridges between the monads 'Dist' (finitely-supported probability distributions) and
--   'LogVec' (finitely-supported NON-normalized log-space measures) -- the ONLY inter-monad
--   structure in the framework:
--
--     decode :: LogVec a -> Dist a          -- softmax a (single-row) leaf      (the READING, per example)
--     encode :: [a] -> Tensor -> LogVec a   -- log a @[B,k]@ probability tensor (the EMBEDDING, batched)
--
--   They are the section-retraction pair @decode . encode = id@ (up to normalization), each realized
--   at the granularity it is actually used: predictions are read out one example at a time ('decode'
--   of the net's logit leaf), observations/inputs are embedded a whole batch at a time ('encode' of a
--   @[B,k]@ one-hot/probability tensor -- e.g. MNIST's observed sums, Binary's certain label). 'encode'
--   floors exact zeros with the affine clamp @(1 - eps) p + eps@ (eps = 1e-13, the inlined
--   'B_Logical.Library.Stable.clampNotZero', so the bridge needs no @B_Logical@ import) so @log@ stays
--   finite. Finiteness is carried by each value's support (the leaf's @[a]@), so these are genuine
--   natural transformations for every @a@ -- no @Sort@/@support@ class is needed.
module A_Categorical.Monads.Bridge
  ( decode,
    encode,
    encDist,
  )
where

import A_Categorical.Monads.Dist (Dist (..))
import A_Categorical.Monads.LogVec (LogVec)
import qualified A_Categorical.Monads.LogVec as LV
import qualified Torch
import qualified Torch.Functional as F

-- | Read a LogVec leaf out as a probability distribution: softmax the (first row of the) log-weights
--   over the leaf's own support. The @Dist@ READING / the readout, per example. Partial: a single
--   'LV.LogLeaf' (the net's logit leaf).
decode :: LogVec a -> Dist a
decode (LV.LogLeaf xs lw) =
  let ps = head (Torch.asValue (F.softmax (F.Dim 1) lw) :: [[Float]])
   in FiniteSupp [(xs !! j, realToFrac (ps !! j)) | j <- [0 .. length xs - 1]]
decode _ = error "Bridge.decode: expected a single LogLeaf (a neural logit leaf)"

-- | Embed a batched distribution into the log world: given a support @xs@ (length @k@) and a per-row
--   probability tensor @probs :: [B,k]@ (a one-hot for a certain observation, or any distribution),
--   the @LogVec@ leaf of log-weights @log((1 - eps) p + eps)@. The @Dist => LogVec@ monad morphism,
--   batched -- the single embedding op the examples use for observations.
encode :: [a] -> Torch.Tensor -> LogVec a
encode xs probs =
  let eps = 1e-13 :: Float
      clamped = (probs `Torch.mul` Torch.asTensor (1 - eps :: Float)) `Torch.add` Torch.asTensor (eps :: Float)
   in LV.LogLeaf xs (Torch.log clamped)

-- | The @Dist => LogVec@ section as a monad morphism (the 'enc' of the section--retraction square):
--   transport the free-monad structure, a finite distribution to its log-weight leaf. On a certain
--   value @eta x = pure x@ it is just @pure x@ in 'LogVec' -- the case used by a two-sided neural
--   symbol  @decode . dig\@LogVec theta . encDist@.
encDist :: Dist a -> LogVec a
encDist (Pure x)        = LV.Pure x
encDist (Bind m k)      = LV.Bind (encDist m) (encDist . k)
encDist (FiniteSupp xs) = encode (map fst xs) (Torch.asTensor [map (realToFrac . snd) xs :: [Float]])
encDist (FinUniform xs) =
  let k = length xs in encode xs (Torch.asTensor [replicate k (1 / fromIntegral k) :: [Float]])
