-- | Category layer for the Binary example: REUSE the library universes — the
--   measure universe @MeasU@ and the geometric universe @GeomU@. To use a
--   different category, define it here instead of re-exporting the library one.
module Examples.Binary.A_Categorical
  ( module Lib.A_Categorical.CategoricalInterpretation,
  )
where

import Lib.A_Categorical.CategoricalInterpretation
