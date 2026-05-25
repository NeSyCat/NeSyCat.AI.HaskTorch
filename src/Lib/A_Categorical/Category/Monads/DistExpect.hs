{-# LANGUAGE GADTs #-}

-- | Expectation under the Dist monad.
--   Dist is always finitely supported, so expectation is a weighted sum.
module Lib.A_Categorical.Category.Monads.DistExpect
  ( distExpect,
    distPTrue,
  )
where

import Lib.A_Categorical.Category.Monads.Dist (Dist (..))

-- | Expectation of @f@ under a finitely-supported distribution.
distExpect :: Dist a -> (a -> Double) -> Double
distExpect (Pure x) f = f x
distExpect (Bind m k) f = distExpect m (\x -> distExpect (k x) f)
distExpect (FiniteSupp xs) f = sum [p * f x | (x, p) <- xs]
distExpect (FinUniform xs) f =
  distExpect (FiniteSupp [(x, 1.0 / fromIntegral (length xs)) | x <- xs]) f

-- | P(True) for Dist: canonical isomorphism Dist(Bool) -> [0,1].
distPTrue :: Dist Bool -> Double
distPTrue m = distExpect m (\b -> if b then 1.0 else 0.0)
