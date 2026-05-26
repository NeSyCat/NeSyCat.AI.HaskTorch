{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}

-- | SIGNATURE TwoMon-BLat -- the "2-monoid bounded lattice": the CONNECTIVE vocabulary
--   (kept separate from the quantifiers, which live in 'A2MonBLat' on top of this).
--   Symbols are grouped Connector (lattice, then the two monoids) then Comparator.
--   Indexed by universe @u@ and truth type @tau@ (fundep @tau -> u@).
module B_Logical.Signature.TwoMonBLat (TwoMonBLat (..)) where

import Data.Kind (Type)

class TwoMonBLat u tau | tau -> u where
  -- | Para PARAMETER SPACE of the logic -- the B-layer analogue of the domain's
  --   horizontal sort \theta (both are the parameter of a parametric morphism).
  --   Default: () (an unparameterised logic).
  type ParamsLogic tau :: Type
  type ParamsLogic tau = ()

  -- == Connector symbols ==

  -- bounded lattice
  top :: tau -- ^ \top
  bot :: tau -- ^ \bot
  neg :: tau -> tau -- ^ \neg
  vee :: ParamsLogic tau -> tau -> tau -> tau -- ^ \vee  (join)
  wedge :: ParamsLogic tau -> tau -> tau -> tau -- ^ \wedge  (meet; default: De Morgan)

  -- the two monoids (units + the residual implication)
  oplus :: tau -> tau -> tau -- ^ \oplus  (additive monoid)
  o0 :: tau -- ^ unit of \oplus
  otimes :: tau -> tau -> tau -- ^ \otimes  (multiplicative monoid)
  o1 :: tau -- ^ unit of \otimes
  implies :: ParamsLogic tau -> tau -> tau -> tau -- ^ \to  (right residual of \otimes; default: \neg a \vee b)

  -- == Comparator symbols ==

  vdash :: tau -> tau -> Bool -- ^ \vdash  (entailment / lattice order)

  -- defaults
  wedge lp x y = neg (vee lp (neg x) (neg y))
  implies lp x y = vee lp (neg x) y
