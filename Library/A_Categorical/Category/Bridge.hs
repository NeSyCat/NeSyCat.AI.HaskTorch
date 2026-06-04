{-# LANGUAGE GADTs #-}

-- | The bridge between the two universe monads: 'Dist' (MeasU, finitely-supported probability
--   distributions) and 'LogVec' (GeomU, finitely-supported NON-normalized log-space measures).
--   Both are genuine endofunctor monads on all of Hask -- the finiteness lives in each value's
--   support (the leaf's @[a]@), not in the carrier type -- so the two bridges below are genuine
--   natural transformations for EVERY @a@: no @Sort@/@support@ class is needed, because the
--   support is carried by the value.
--
--     decode :: LogVec a -> Dist a   -- softmax / normalize  (the MeasU READING of a computation)
--     encode :: Dist a   -> LogVec a -- log of the masses    (the inverse embedding)
--
--   They form a section-retraction: @decode . encode = id@ (and @encode . decode = normalize@).
--   A neural symbol's MeasU reading is just @decode@ of its GeomU leaf, so the support is written
--   ONCE, at the GeomU construction site (e.g. @digit \@GeomU theta = LogLeaf [0..9] . cnn theta@,
--   @digit \@MeasU theta = decode . digit \@GeomU theta@). This is the modular "softmax/decoder"
--   that used to be the awkwardly-named @categorical@.
module A_Categorical.Category.Bridge
  ( decode,
    encode,
  )
where

import A_Categorical.Category.Monads.Dist (Dist (..))
import A_Categorical.Category.Monads.LogVec (LogVec)
import qualified A_Categorical.Category.Monads.LogVec as LV
import qualified Torch
import qualified Torch.Functional as F

-- | Read a LogVec leaf out as a probability distribution: softmax the (row of) log-weights over
--   the leaf's own support. The MeasU reading / the readout. Defined for any @a@ (the support
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
  LV.LogLeaf (map fst xs) (Torch.asTensor ([[realToFrac (log p) | (_, p) <- xs]] :: [[Float]]))
encode (FinUniform xs) =
  encode (FiniteSupp [(x, 1.0 / fromIntegral (length xs)) | x <- xs])
encode (Bind m k) = LV.Bind (encode m) (encode . k)
