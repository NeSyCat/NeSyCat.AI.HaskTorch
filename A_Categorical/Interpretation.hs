{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

-- | Star interpretation: assigns concrete monads and categories
--   to the abstract roles declared in StarTheory.
--
--   Provides the two concrete universes:
--     GeomU : geometry paradigm (tensors + Identity)
--     MeasU : measure theory paradigm (data + Dist)
module A_Categorical.Interpretation
  ( GeomU,
    MeasU,
  )
where

import A_Categorical.Theory (Universe (..))
import A_Categorical.Category.Categories.Data (DataObj)
import A_Categorical.Category.Categories.Tens (TensObj)
import A_Categorical.Category.Monads.Dist (Dist)
import Data.Functor.Identity (Identity)

-- | Geometry paradigm: tensors + Identity monad.
data GeomU

-- | Measure theory paradigm: data types + Dist monad.
data MeasU

instance Universe GeomU where
  type Cat GeomU = TensObj
  type M GeomU = Identity

instance Universe MeasU where
  type Cat MeasU = DataObj
  type M MeasU = Dist
