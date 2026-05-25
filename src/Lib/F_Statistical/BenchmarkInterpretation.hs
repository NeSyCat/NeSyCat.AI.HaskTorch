{-# LANGUAGE TypeFamilies #-}

-- | DATA benchmark interpretation: composes each metric symbol of
--   'BenchmarkSignature' from the library primitives.
--
--   Prediction = Double (expectation of Dist Bool from classifierA @MeasU)
--   Label      = Bool   (from labelA @MeasU)
module Lib.F_Statistical.BenchmarkInterpretation () where

import Lib.F_Statistical.BenchmarkSignature (BenchmarkSignature (..))
import Lib.F_Statistical.Library.FractionTrue (fractionTrue)
import Lib.F_Statistical.Library.HarmonicMean (harmonicMean)
import Lib.F_Statistical.Library.MeanWhere (meanWhere)
import Lib.F_Statistical.Library.Threshold (threshold)

instance BenchmarkSignature Double where
  type MetricVal Double = Double

  accuracy pairs =
    let correct = [threshold p 0.5 == l | (p, l) <- pairs]
     in fractionTrue correct

  precision pairs =
    let predicted = [(p, l) | (p, l) <- pairs, threshold p 0.5]
     in if null predicted
          then 0.0
          else fractionTrue [l | (_, l) <- predicted]

  recall pairs =
    let positives = [(p, l) | (p, l) <- pairs, l]
     in if null positives
          then 0.0
          else fractionTrue [threshold p 0.5 | (p, _) <- positives]

  f1Score pairs = harmonicMean (precision pairs) (recall pairs)

  confidence pairs =
    let preds = map fst pairs
        labels = map snd pairs
     in (meanWhere preds labels, meanWhere preds (map not labels))
