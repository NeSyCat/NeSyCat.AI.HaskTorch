{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Logical interpretation: Classical Boolean Logic ($\Omega = \{\text{True}, \text{False}\}$)
--
--   This module provides the interpretation of LogicalSignature and LogicalQuantSignature
--   in the MeasU universe with Omega = Bool.
module B_Logical.Interpretations.Boolean
  ( Omega,
    -- * Re-exported typeclass interface
    module B_Logical.LogicalSignature,
    module B_Logical.LogicalQuantSignature,
    -- * Comparison predicates
    (.==),
    (./=),
    (.<),
    (.>),
    (.<=),
    (.>=),
    b2o,
  )
where

import A_Categorical.CategoricalInterpretation (MeasU)
import A_Categorical.Category.Monads.Dist ()  -- Monad instance for Dist
import B_Logical.LogicalQuantSignature (LogicalQuantSignature (..), Guard)
import B_Logical.LogicalSignature (LogicalSignature (..))


infix 4 .==, ./=, .<, .>, .<=, .>=

-- | Omega := I(tau) = {True, False}
type Omega = Bool

------------------------------------------------------
-- LogicalSignature instance: Boolean lattice operations
------------------------------------------------------

instance LogicalSignature MeasU Bool where
  type ParamsLogic Bool = ()
  vdash = (<=)
  vee _ = (||)
  wedge _ = (&&)
  bot = False
  top = True
  neg = not
  implies _ a b = not a || b
  oplus = (||)
  otimes = (&&)
  v0 = False
  v1 = True

------------------------------------------------------
-- Guard type instance: MeasU guards are finite subsets (lists)
------------------------------------------------------

type instance Guard MeasU a = [a]

------------------------------------------------------
-- LogicalQuantSignature: one generic instance for all point types
--   Guard MeasU a = [a], so guard :: [a] and phi :: a -> Dist Bool.
--   bigWedge/bigVee = commutator (mapM) then lattice reduce (inf/sup)
--   bigOplus/bigOtimes = commutator then measure reduce
------------------------------------------------------

instance LogicalQuantSignature a MeasU Bool where
  -- forall = commutator + inf (lattice meet = and)
  bigWedge _ guard phi = do
    omegas <- mapM phi guard       -- commutator: (M Omega)^A -> M(Omega^A)
    return (foldl (wedge ()) True omegas)  -- lattice inf via wedge
  -- exists = commutator + sup (lattice join = or)
  bigVee _ guard phi = do
    omegas <- mapM phi guard       -- commutator
    return (foldl (vee ()) False omegas)   -- lattice sup via vee
  -- oplus/otimes = commutator + measure quantifier (expectation-based)
  bigOplus guard phi = bigVee () guard phi
  bigOtimes guard phi = bigWedge () guard phi

------------------------------------------------------
-- Comparison predicates
------------------------------------------------------

(.==) :: (Eq a) => a -> a -> Omega
x .== y = x == y

(.<) :: (Ord a) => a -> a -> Omega
x .< y = x < y

(.>) :: (Ord a) => a -> a -> Omega
x .> y = x > y

(.<=) :: (Ord a) => a -> a -> Omega
x .<= y = x <= y

(.>=) :: (Ord a) => a -> a -> Omega
x .>= y = x >= y

(./=) :: (Eq a) => a -> a -> Omega
x ./= y = x /= y

b2o :: Bool -> Omega
b2o = id
