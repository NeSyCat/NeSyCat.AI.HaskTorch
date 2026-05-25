-- | Category layer for the Binary example: REUSE the library universes — the
--   measure universe @MeasU@ and the geometric universe @GeomU@. To use a
--   different category, define it here instead of re-exporting the library one.
module Binary.A_Categorical
  ( module A_Categorical.CategoricalInterpretation,
  )
where

import A_Categorical.CategoricalInterpretation
