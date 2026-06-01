{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | SIGNATURE of classical logic: a bounded lattice with complement, PLUS its
--   quantifiers -- one vocabulary covering connectives AND aggregations together
--   (you always use a logic as a whole). A concrete interpretation (e.g. Boolean,
--   \tau = Bool) is an instance defining all of it in one file. Indexed by universe
--   @u@ and truth type @tau@ (fundep @tau -> u@; use newtypes to reuse a carrier).
module B_Logical.Signature.Classical (ClassicalSignature (..)) where

import A_Categorical.CategoricalSignature (Framework (..))
import B_Logical.Signature.Guard (Guard)

class (Framework u, Monad (M u)) => ClassicalSignature u tau | tau -> u where
  -- connectives
  top :: tau -- ^ \top  (true / lattice greatest)
  bot :: tau -- ^ \bot  (false / lattice least)
  neg :: tau -> tau -- ^ \neg  (complement)
  vee :: tau -> tau -> tau -- ^ \vee  (join)
  wedge :: tau -> tau -> tau -- ^ \wedge  (meet)
  implies :: tau -> tau -> tau -- ^ \to  (material implication)
  -- quantifiers (aggregate the connectives over a guard; @a@ = the point type)
  bigVee :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigvee  (\exists)
  bigWedge :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigwedge  (\forall)

  -- Standard defaults (override per interpretation if a primitive is cheaper):
  wedge x y = neg (vee (neg x) (neg y)) -- De Morgan
  implies x y = vee (neg x) y -- material implication
