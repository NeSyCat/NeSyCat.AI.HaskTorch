{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module Lib.B_Logical.LogicalQuantSignature (
    LogicalQuantSignature (..),
    Guard,
)
where

import Lib.A_Categorical.CategoricalSignature (Universe (..))
import Lib.B_Logical.LogicalSignature (LogicalSignature (..))
import Data.Kind (Type)

-- | Guard: the subset that a guarded quantifier ranges over.
type family Guard u a :: Type

-- | Theory of an aggregated 2-monoid bounded lattice (A2Mon-BLat).
class
    (LogicalSignature u tau, Universe u, Monad (M u)) =>
    LogicalQuantSignature a u tau
    where
    bigWedge :: ParamsLogic tau -> Guard u a -> (a -> M u tau) -> M u tau
    bigVee :: ParamsLogic tau -> Guard u a -> (a -> M u tau) -> M u tau
    bigOplus :: Guard u a -> (a -> M u tau) -> M u tau
    bigOtimes :: Guard u a -> (a -> M u tau) -> M u tau
