{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Logical interpretation: classical Boolean logic (Omega = \{True, False\}) in the
--   MeasU universe. Instantiates 'TwoMonBLat' (the connectives) and 'A2MonBLat' (the
--   quantifiers).
module B_Logical.Interpretations.Boolean
  ( Omega,
    -- * Re-exported signature interface
    module B_Logical.Signature.TwoMonBLat,
    module B_Logical.Signature.A2MonBLat,
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
import A_Categorical.Category.Monads.Dist () -- Monad instance for Dist
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))

infix 4 .==, ./=, .<, .>, .<=, .>=

-- | Omega := I(tau) = \{True, False\}
type Omega = Bool

-- | MeasU guards are finite subsets (lists).
type instance Guard MeasU a = [a]

------------------------------------------------------
-- TwoMonBLat: the connective interpretation (Boolean lattice + the two monoids)
------------------------------------------------------

instance TwoMonBLat MeasU Bool where
  type ParamsLogic Bool = ()
  -- bounded lattice
  top = True
  bot = False
  neg = not
  vee _ = (||)
  wedge _ = (&&)
  -- the two monoids (units + residual)
  oplus = (||)
  o0 = False
  otimes = (&&)
  o1 = True
  implies _ a b = not a || b
  -- comparator
  vdash = (<=)

------------------------------------------------------
-- A2MonBLat: the quantifier interpretation (commutator via mapM, then lattice reduce)
------------------------------------------------------

instance A2MonBLat a MeasU Bool where
  -- forall = commutator + inf (lattice meet = and)
  bigWedge _ guard phi = do
    omegas <- mapM phi guard
    return (foldl (wedge ()) True omegas)
  -- exists = commutator + sup (lattice join = or)
  bigVee _ guard phi = do
    omegas <- mapM phi guard
    return (foldl (vee ()) False omegas)
  -- the monoid aggregations reduce to the lattice ones in the Boolean model
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
