{-# LANGUAGE TypeFamilies #-}

-- | Reusable benchmark reporting. A 'Report' is just a list of labeled metrics,
--   so ANY example reports whatever metrics make sense for it (binary
--   classification, multi-class sum/digit accuracy, ...) without cramming into
--   fixed fields. Printing and N-run averaging are generic. The binary-metric
--   helpers ('evaluate', 'runMetrics') remain here for examples that want them.
module G_Statistical.Report
  ( Report (..),
    printReport,
    averageReports,
    summarizeReports,
    printSummary,
    meanStd,
    runAverage,
    evaluate,
    runMetrics,
  )
where

import Control.Monad (forM)
import Data.List (intercalate)
import G_Statistical.BenchmarkInterpretation ()
import G_Statistical.BenchmarkSignature (BenchmarkSignature (..))
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

-- | Sample mean and (n-1) standard deviation -- the ONE reusable stat behind every
--   "mean +/- std" the framework prints (seed-averaged metrics here; reusable for step-timing).
meanStd :: [Double] -> (Double, Double)
meanStd [] = (0, 0)
meanStd xs = (m, s)
  where
    nn = fromIntegral (length xs)
    m = sum xs / nn
    s
      | length xs < 2 = 0
      | otherwise = sqrt (sum [(x - m) ^ (2 :: Int) | x <- xs] / (nn - 1))

-- | Per-metric @(label, mean, std)@ over several reports, matched by metric position.
summarizeReports :: [Report] -> [(String, Double, Double)]
summarizeReports [] = []
summarizeReports rs@(Report ms0 : _) =
  [ (k, m, s)
  | (i, (k, _)) <- zip [0 ..] ms0,
    let (m, s) = meanStd [snd (ms !! i) | Report ms <- rs]
  ]

-- | Print a @label: mean +/- std@ block under a title.
printSummary :: String -> [(String, Double, Double)] -> IO ()
printSummary title rows = do
  putStrLn ""
  putStrLn (title ++ ":")
  mapM_ (\(k, m, s) -> printf "  %-16s %.4f +/- %.4f\n" (k ++ ":") m s) rows

-- | Run an experiment @n@ times: for n=1 print the single report, else a compact
--   per-run line then the field-wise mean +/- std (the seed-averaged report).
runAverage :: String -> Int -> IO Report -> IO ()
runAverage title 1 experiment = experiment >>= printReport title
runAverage title n experiment = do
  printf "Running %d runs...\n" n
  reports <- forM [1 .. n] $ \i -> do
    r <- experiment
    printf "  Run %2d:  %s\n" (i :: Int) (summaryLine r)
    return r
  printSummary (title ++ printf " - mean +/- std over %d runs" n) (summarizeReports reports)
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
