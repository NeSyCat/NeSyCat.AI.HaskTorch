{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}

-- | The Giry monad: general probability measures (finite, countable, continuous),
--   together with its free-monad realization (symbolic/lazy).
--   Evaluation lives in A_Categorical/Category/Monads/GiryExpect.
module A_Categorical.Category.Monads.Giry
  ( Giry (..),
  )
where

import Control.Monad (ap)
import Statistics.Distribution (ContDistr, Mean, Variance)

-- | The Giry Monad: general probability measures (finite, countable, continuous).
--   Also a free monad -- evaluation via expect.
data Giry a where
  -- Monadic structure
  GPure :: a -> Giry a
  GBind :: Giry x -> (x -> Giry a) -> Giry a
  -- Finitely supported
  GFiniteSupp :: [(a, Double)] -> Giry a
  GFinUniform :: [a] -> Giry a
  -- Countably infinite support
  Poisson :: Double -> Giry Int
  Geometric :: Double -> Giry Int
  -- Continuous (over Reals)
  Normal :: Double -> Double -> Giry Double
  Uniform :: Double -> Double -> Giry Double
  -- and more standard distributions...
  Exponential :: Double -> Giry Double
  Beta :: Double -> Double -> Giry Double
  Gamma :: Double -> Double -> Giry Double
  Laplace :: Double -> Double -> Giry Double
  StudentT :: Double -> Giry Double
  GenericCont :: (ContDistr d, Mean d, Variance d) => d -> Giry Double
  ContinuousPdf :: (Double -> Double) -> (Double, Double) -> Giry Double

instance Functor Giry where
  fmap :: (a -> b) -> Giry a -> Giry b
  fmap f m = GBind m (GPure . f)

instance Applicative Giry where
  pure :: a -> Giry a
  pure = GPure

  (<*>) :: Giry (a -> b) -> Giry a -> Giry b
  (<*>) = ap

instance Monad Giry where
  return :: a -> Giry a
  return = pure

  (>>=) :: Giry a -> (a -> Giry b) -> Giry b
  (>>=) = GBind
