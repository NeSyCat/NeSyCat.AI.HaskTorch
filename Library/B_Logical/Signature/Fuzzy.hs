{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | SIGNATURE of fuzzy (t-norm) logic on truth degrees in [0,1] -- a residuated
--   lattice -- with its quantifiers, connectives AND aggregations in one vocabulary.
--   The three standard interpretations (Goedel, Lukasiewicz, Product) are instances,
--   each in ONE file; they differ on the STRONG \otimes / \oplus and the implication
--   \to, while the lattice \wedge / \vee (min / max) and \neg are usually shared.
module B_Logical.Signature.Fuzzy (FuzzySignature (..)) where

import A_Categorical.CategoricalSignature (Universe (..))
import B_Logical.Signature.Guard (Guard)

class (Universe u, Monad (M u)) => FuzzySignature u tau | tau -> u where
  -- connectives
  top :: tau -- ^ \top  (truth degree 1)
  bot :: tau -- ^ \bot  (truth degree 0)
  neg :: tau -> tau -- ^ standard negation (1 - x)
  wedge :: tau -> tau -> tau -- ^ \wedge  weak conjunction (lattice meet = min)
  vee :: tau -> tau -> tau -- ^ \vee  weak disjunction (lattice join = max)
  otimes :: tau -> tau -> tau -- ^ \otimes  strong conjunction (the t-norm)
  oplus :: tau -> tau -> tau -- ^ \oplus  strong disjunction (the t-conorm)
  implies :: tau -> tau -> tau -- ^ \to  residuated implication (right adjoint to \otimes)
  -- quantifiers
  bigWedge :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigwedge  (\forall)
  bigVee :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigvee  (\exists)
  bigOplus :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigoplus  (additive aggregation)
  bigOtimes :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigotimes  (multiplicative aggregation)
