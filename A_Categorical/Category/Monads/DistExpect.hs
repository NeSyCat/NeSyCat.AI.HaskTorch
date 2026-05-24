{-# LANGUAGE GADTs #-}

-- | Expectation under the Dist monad.
--   Dist is always finitely supported, so expectation is a weighted sum.
module B_Logical.ExpectDist
  ( expectDist,
    pTrueDist,
  )
where

import A_Categorical.Category.Monads.Dist (Dist (..))

-- | Expectation of @f@ under a finitely-supported distribution.
expectDist :: Dist a -> (a -> Double) -> Double
expectDist (Pure x) f = f x
expectDist (Bind m k) f = expectDist m (\x -> expectDist (k x) f)
expectDist (FiniteSupp xs) f = sum [p * f x | (x, p) <- xs]
expectDist (FinUniform xs) f =
  expectDist (FiniteSupp [(x, 1.0 / fromIntegral (length xs)) | x <- xs]) f

-- | P(True) for Dist: canonical isomorphism Dist(Bool) -> [0,1].
pTrueDist :: Dist Bool -> Double
pTrueDist m = expectDist m (\b -> if b then 1.0 else 0.0)
