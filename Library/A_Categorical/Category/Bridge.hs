{-# LANGUAGE GADTs #-}

-- | The bridge between the two monads: 'Dist' (finitely-supported probability distributions) and
--   'LogVec' (finitely-supported NON-normalized log-space measures). Both are genuine endofunctor
--   monads on all of Hask -- the finiteness lives in each value's support (the leaf's @[a]@), not
--   in the carrier type -- so the bridges below are genuine natural transformations for EVERY @a@:
--   no @Sort@/@support@ class is needed, because the support is carried by the value.
--
--     decode      :: LogVec a -> Dist a   -- softmax / normalize  (the READING of a computation)
--     encode      :: Dist a   -> LogVec a -- log of the masses    (the embedding)
--     encodeBatch :: [a] -> Tensor -> LogVec a  -- the BATCHED encode: a @[B,k]@ row of
--                                               -- probabilities + an explicit support -> a leaf
--
--   'decode'/'encode' form a section-retraction: @decode . encode = id@ (and
--   @encode . decode = normalize@). A neural symbol's @Dist@ reading is just @decode@ of its
--   @LogVec@ leaf, so the support is written ONCE, at the @LogVec@ construction site (e.g.
--   @digit \@LogVec theta = LogLeaf [0..9] . cnn theta@, @digit \@Dist theta = decode . digit
--   \@LogVec theta@). 'encodeBatch' is the SINGLE embedding op for an observation/input given as a
--   batched probability (or one-hot) tensor -- the batched realization of @encode@ that the
--   examples' observation leaves (MNIST's @obsLeaf@, Binary's certain @labelA@ leaf) call directly.
module A_Categorical.Category.Bridge
  ( decode,
    encode,
    encodeBatch,
  )
where

import A_Categorical.Category.Monads.Dist (Dist (..))
import A_Categorical.Category.Monads.LogVec (LogVec)
import qualified A_Categorical.Category.Monads.LogVec as LV
import qualified Torch
import qualified Torch.Functional as F

-- | Read a LogVec leaf out as a probability distribution: softmax the (row of) log-weights over
--   the leaf's own support. The @Dist@ reading / the readout. Defined for any @a@ (the support
--   @xs@ is carried by the leaf). Partial: a single 'LV.LogLeaf' (the net's logit leaf), like
--   'A_Categorical.Category.Monads.LogVecExpect.logVecLeafTensor'.
decode :: LogVec a -> Dist a
decode (LV.LogLeaf xs lw) =
  let ps = head (Torch.asValue (F.softmax (F.Dim 1) lw) :: [[Float]])
   in FiniteSupp [(xs !! j, realToFrac (ps !! j)) | j <- [0 .. length xs - 1]]
decode _ = error "Bridge.decode: expected a single LogLeaf (a neural logit leaf)"

-- | Embed a probability distribution into the log world: log of its masses. The inverse of
--   'decode' (a monad morphism @Dist => LogVec@; @decode . encode = id@). Defined for any @a@.
encode :: Dist a -> LogVec a
encode (Pure x) = LV.Pure x
encode (FiniteSupp xs) =
  encodeBatch (map fst xs) (Torch.asTensor ([[realToFrac p | (_, p) <- xs]] :: [[Float]]))
encode (FinUniform xs) =
  encode (FiniteSupp [(x, 1.0 / fromIntegral (length xs)) | x <- xs])
encode (Bind m k) = LV.Bind (encode m) (encode . k)

-- | The BATCHED 'encode': given a support @xs@ (length @k@) and a per-row probability tensor
--   @probs :: [B,k]@ (a one-hot for a certain observation, or any distribution), return the
--   corresponding @LogVec@ leaf -- log-weights @log((1 - eps) p + eps)@ (the affine clamp keeps
--   exact zeros finite; @eps = 1e-13@, matching @B_Logical.Library.Stable.clampNotZero@, inlined
--   here so the bridge needs no @B_Logical@ import). This is the single embedding op the examples
--   use for observations (MNIST's @obsLeaf@, Binary's certain @labelA@ leaf).
encodeBatch :: [a] -> Torch.Tensor -> LogVec a
encodeBatch xs probs =
  let eps = 1e-13 :: Float
      clamped = (probs `Torch.mul` Torch.asTensor (1 - eps :: Float)) `Torch.add` Torch.asTensor (eps :: Float)
   in LV.LogLeaf xs (Torch.log clamped)
