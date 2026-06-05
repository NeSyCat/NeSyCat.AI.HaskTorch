{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Logical interpretation: classical Boolean logic (Omega = \{True, False\}) in the
--   @Dist@ monad. Instantiates 'TwoMonBLat' (the connectives) and 'A2MonBLat' (the
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

import A_Categorical.Category.Monads.Dist (Dist) -- the monad (+ its Monad instance)
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))

infix 4 .==, ./=, .<, .>, .<=, .>=

-- | Omega := I(tau) = \{True, False\}
type Omega = Bool

-- | @Dist@ guards are finite subsets (lists).
type instance Guard Dist a = [a]

------------------------------------------------------
-- TwoMonBLat: the connective interpretation (Boolean lattice + the two monoids). Universe-free
-- -- this crisp @Bool@ truth algebra is shared by every interpretation (the @Dist Bool@
-- and @LogVec Bool@ readings); only the quantifiers ('A2MonBLat', below for Dist) pick a monad.
------------------------------------------------------

instance TwoMonBLat Bool where
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

instance A2MonBLat a Dist Bool where
  -- forall = commutator + inf (lattice meet = and)
  bigWedge _ guard phi = do
    omegas <- mapM phi guard
    return (foldl (wedge ()) True omegas)
  -- exists = commutator + sup (lattice join = or)
  bigVee _ guard phi = do
    omegas <- mapM phi guard
    return (foldl (vee ()) False omegas)
  -- the monoid aggregations reduce to the lattice ones in the Boolean model (inlined rather
  -- than delegating to bigVee/bigWedge: the self-call's monad is no longer fixed by the
  -- truth type, and here it is plainly Dist).
  bigOplus guard phi = do
    omegas <- mapM phi guard
    return (foldl (vee ()) False omegas)
  bigOtimes guard phi = do
    omegas <- mapM phi guard
    return (foldl (wedge ()) True omegas)

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
