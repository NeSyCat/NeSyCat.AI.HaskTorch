{-# LANGUAGE TypeFamilies #-}

-- | Reusable benchmark reporting. Domain-agnostic: any benchmark example reuses
--   'evaluate' (build prediction/label pairs), 'runMetrics' (collect the metrics),
--   'printReport' (print), and 'runAverage' (run N times and average natively) --
--   so a new benchmark recodes none of this, and no external script is needed.
module F_Statistical.Report
  ( BenchmarkReport (..),
    evaluate,
    runMetrics,
    printReport,
    average,
    runAverage,
  )
where

import Control.Monad (forM)
import F_Statistical.BenchmarkInterpretation ()
import F_Statistical.BenchmarkSignature (BenchmarkSignature (..))
import Text.Printf (printf)

-- | The standard report: metrics on the train and test splits.
data BenchmarkReport = BenchmarkReport
  { reportAccTrain :: Double,
    reportAccTest :: Double,
    reportF1 :: Double,
    reportPrecision :: Double,
    reportConfPos :: Double,
    reportConfNeg :: Double
  }

-- | Apply a model and a labeller pointwise to build (prediction, label) pairs.
evaluate :: (pt -> pred) -> (pt -> Bool) -> [pt] -> [(pred, Bool)]
evaluate predict label = map (\pt -> (predict pt, label pt))

-- | Collect the standard metrics from train/test (prediction, label) pairs.
runMetrics :: [(Double, Bool)] -> [(Double, Bool)] -> BenchmarkReport
runMetrics trainPairs testPairs =
  let (pPos, pNeg) = confidence testPairs
   in BenchmarkReport
        { reportAccTrain = accuracy trainPairs,
          reportAccTest = accuracy testPairs,
          reportF1 = f1Score testPairs,
          reportPrecision = precision testPairs,
          reportConfPos = pPos,
          reportConfNeg = pNeg
        }

-- | Print a benchmark report under a title.
printReport :: String -> BenchmarkReport -> IO ()
printReport title r = do
  putStrLn ""
  putStrLn (title ++ ":")
  printf "  Accuracy:    Train=%.4f  Test=%.4f\n" (reportAccTrain r) (reportAccTest r)
  printf "  F1 Score:    %.4f\n" (reportF1 r)
  printf "  Precision:   %.4f\n" (reportPrecision r)
  printf "  Confidence:  P+=%.4f  P-=%.4f\n" (reportConfPos r) (reportConfNeg r)

-- | Field-wise mean of several reports.
average :: [BenchmarkReport] -> BenchmarkReport
average rs =
  BenchmarkReport
    (mean reportAccTrain)
    (mean reportAccTest)
    (mean reportF1)
    (mean reportPrecision)
    (mean reportConfPos)
    (mean reportConfNeg)
  where
    n = fromIntegral (max 1 (length rs))
    mean f = sum (map f rs) / n

-- | Run a benchmark experiment @n@ times and report. For @n == 1@ just print the
--   single report; otherwise print a compact line per run, then the average.
runAverage :: String -> Int -> IO BenchmarkReport -> IO ()
runAverage title 1 experiment = experiment >>= printReport title
runAverage title n experiment = do
  printf "Running %d runs...\n" n
  reports <- forM [1 .. n] $ \i -> do
    r <- experiment
    printf "  Run %2d:  Acc(test)=%.4f  F1=%.4f  Prec=%.4f\n"
      (i :: Int) (reportAccTest r) (reportF1 r) (reportPrecision r)
    return r
  printReport (title ++ (printf " - average over %d runs" n :: String)) (average reports)
