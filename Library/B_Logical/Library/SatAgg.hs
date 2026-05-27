{-# LANGUAGE FlexibleContexts #-}

-- | SatAgg — the satisfaction of a whole knowledge base: the CONJUNCTION of its closed
--   formulas (each already fully quantified). Built from the strong-conjunction monoid of
--   the chosen logic (@otimes@ with unit @o1@), so it is interpreted per logic just like
--   any other connective. A single-formula KB collapses to that formula (@o1@ is the unit).
--
--   This is the @\bigotimes_{phi \in KB} phi@ that the grammatical layer (D) exports as
--   @sat@; the inference layer then only penalizes it (@lossKnow . sat@).
module B_Logical.Library.SatAgg (satAgg) where

import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))

-- | Conjoin the closed-formula satisfactions of a knowledge base: @foldr otimes o1@.
satAgg :: (TwoMonBLat u tau) => [tau] -> tau
satAgg = foldr otimes o1
