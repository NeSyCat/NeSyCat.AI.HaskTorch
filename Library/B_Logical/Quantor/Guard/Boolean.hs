{-# LANGUAGE TypeFamilies #-}

-- | Guard type for the MeasU universe: finite lists.
--
-- A guard for classical Boolean quantification is a finite subset of the
-- domain, represented as a list.  Ranging over Guard MeasU a = [a] gives
-- the usual universal/existential quantifiers over finite domains.
module B_Logical.Quantor.Guard.Boolean where

import A_Categorical.CategoricalSignature (Universe (..))
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.LogicalQuantSignature (Guard)

-- | Guard MeasU a = [a]  (finite subset as list)
type instance Guard MeasU a = [a]
