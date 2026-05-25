{-# LANGUAGE TypeFamilies #-}

-- | Reusable benchmark reporting. A 'Report' is just a list of labeled metrics,
--   so ANY example reports whatever metrics make sense for it (binary
--   classification, multi-class sum/digit accuracy, ...) without cramming into
--   fixed fields. Printing and N-run averaging are generic. The binary-metric
--   helpers ('evaluate', 'runMetrics') remain here for examples that want them.
module Lib.F_Statistical.Report
  ( Report (..),
    printReport,
    averageReports,
    runAverage,
    evaluate,
    runMetrics,
  )
where

import Control.Monad (forM)
import Data.List (intercalate)
import Lib.F_Statistical.BenchmarkInterpretation ()
import Lib.F_Statistical.BenchmarkSignature (BenchmarkSignature (..))
import Text.Printf (printf)

-- | A flexible report: ordered, labeled metrics. Each example builds its own.
newtype Report = Report {reportMetrics :: [(String, Double)]}

-- | Print a report under a title (one @label: value@ line per metric).
printReport :: String -> Report -> IO ()
printReport title (Report ms) = do
  putStrLn ""
  putStrLn (title ++ ":")
  mapM_ (\(k, v) -> printf "  %-16s %.4f\n" (k ++ ":") v) ms

-- | Mean of several reports, matched by metric position (labels from the first).
averageReports :: [Report] -> Report
averageReports [] = Report []
averageReports rs@(Report ms0 : _) =
  Report [(k, sum [snd (m !! i) | Report m <- rs] / n) | (i, (k, _)) <- zip [0 ..] ms0]
  where
    n = fromIntegral (length rs)

-- | Run an experiment @n@ times: for n=1 print the single report, else a compact
--   per-run line then the field-wise average.
runAverage :: String -> Int -> IO Report -> IO ()
runAverage title 1 experiment = experiment >>= printReport title
runAverage title n experiment = do
  printf "Running %d runs...\n" n
  reports <- forM [1 .. n] $ \i -> do
    r <- experiment
    printf "  Run %2d:  %s\n" (i :: Int) (summaryLine r)
    return r
  printReport (title ++ printf " - average over %d runs" n) (averageReports reports)
  where
    summaryLine (Report ms) = intercalate "  " [printf "%s=%.4f" k v | (k, v) <- ms]

-- | Apply a model and a labeller pointwise to build (prediction, label) pairs.
evaluate :: (pt -> pred) -> (pt -> Bool) -> [pt] -> [(pred, Bool)]
evaluate predict label = map (\pt -> (predict pt, label pt))

-- | The standard binary-classification metrics, as a labeled 'Report'.
runMetrics :: [(Double, Bool)] -> [(Double, Bool)] -> Report
runMetrics trainPairs testPairs =
  let (pPos, pNeg) = confidence testPairs
   in Report
        [ ("Acc(train)", accuracy trainPairs),
          ("Acc(test)", accuracy testPairs),
          ("F1", f1Score testPairs),
          ("Precision", precision testPairs),
          ("Conf+", pPos),
          ("Conf-", pNeg)
        ]
