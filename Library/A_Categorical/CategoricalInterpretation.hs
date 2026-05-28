{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

-- | Star interpretation: assigns concrete monads and categories
--   to the abstract roles declared in StarTheory.
--
--   Provides the two concrete universes:
--     GeomU : geometry paradigm (tensors + LogVec)
--     MeasU : measure theory paradigm (data + Dist)
module A_Categorical.CategoricalInterpretation
  ( GeomU,
    MeasU,
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import A_Categorical.Category.Categories.Data (DataObj)
import A_Categorical.Category.Categories.Tens (TensObj)
import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.Category.Monads.LogVec (LogVec)

-- | Geometry paradigm: tensors + LogVec monad (its bind is the log-space convolution).
data GeomU

-- | Measure theory paradigm: data types + Dist monad.
data MeasU

instance Universe GeomU where
  type Cat GeomU = TensObj
  type M GeomU = LogVec

instance Universe MeasU where
  type Cat MeasU = DataObj
  type M MeasU = Dist
