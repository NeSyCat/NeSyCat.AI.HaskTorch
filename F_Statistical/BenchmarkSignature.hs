{-# LANGUAGE TypeFamilies #-}

-- | The benchmark signature (level zeta): the symbols to evaluate a trained
--   classifier against ground truth.
--
--   Sort symbol:
--     MetricVal                                  the metric values (R)
--   Function symbols:
--     accuracy   : [(pred, label)] -> MetricVal              fraction correct
--     f1Score    : [(pred, label)] -> MetricVal              harmonic mean of precision/recall
--     precision  : [(pred, label)] -> MetricVal              tp / (tp + fp)
--     recall     : [(pred, label)] -> MetricVal              tp / (tp + fn)
--     confidence : [(pred, label)] -> (MetricVal, MetricVal) mean P+ and P-
--
--   Concrete metric primitives live in F_Statistical/Library; an interpretation
--   composes the metrics from them. Label is Bool (binary classification).
module F_Statistical.BenchmarkSignature
  ( BenchmarkSignature (..),
  )
where

import Data.Kind (Type)

-- | The benchmark signature: the sort symbol @MetricVal@ together with the
--   metric function symbols. An interpretation (instance) assigns @MetricVal@ a
--   concrete sort and each function symbol a concrete morphism.
class BenchmarkSignature pred where
  -- | Sort symbol: the type of metric values (R).
  type MetricVal pred :: Type

  -- | accuracy : fraction of predictions matching ground truth.
  accuracy :: [(pred, Bool)] -> MetricVal pred

  -- | f1Score : harmonic mean of precision and recall.
  f1Score :: [(pred, Bool)] -> MetricVal pred

  -- | precision : tp / (tp + fp).
  precision :: [(pred, Bool)] -> MetricVal pred

  -- | recall : tp / (tp + fn).
  recall :: [(pred, Bool)] -> MetricVal pred

  -- | confidence : (mean probability for positive samples, mean for negative).
  confidence :: [(pred, Bool)] -> (MetricVal pred, MetricVal pred)
