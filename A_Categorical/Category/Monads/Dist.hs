{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}

-- | The Dist monad: finitely supported probability distributions,
--   together with its free-monad realization (symbolic/lazy).
--   Evaluation lives in B_Logical/DA_Realization/ExpectDist.
module A_Categorical.Category.Monads.Dist
  ( Dist (..),
  )
where

import Control.Monad (ap)

-- | The Dist Monad: finitely supported probability distributions.
--   Represented as a free monad (symbolic/lazy) -- evaluation via expect.
--   Only finite support constructors are allowed.
data Dist a where
  Pure :: a -> Dist a
  Bind :: Dist x -> (x -> Dist a) -> Dist a
  FiniteSupp :: [(a, Double)] -> Dist a
  FinUniform :: [a] -> Dist a

instance Functor Dist where
  fmap :: (a -> b) -> Dist a -> Dist b
  fmap f m = Bind m (Pure . f)

instance Applicative Dist where
  pure :: a -> Dist a
  pure = Pure

  (<*>) :: Dist (a -> b) -> Dist a -> Dist b
  (<*>) = ap

instance Monad Dist where
  return :: a -> Dist a
  return = pure

  (>>=) :: Dist a -> (a -> Dist b) -> Dist b
  (>>=) = Bind
