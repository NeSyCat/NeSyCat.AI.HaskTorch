{-# LANGUAGE TypeFamilies #-}

-- | Reusable benchmark reporting. Domain-agnostic: any benchmark example reuses
--   'evaluate' (build prediction/label pairs), 'runMetrics' (collect the metrics)
--   and 'printReport' (print) — so a new benchmark recodes none of this.
module F_Statistical.Report
  ( BenchmarkReport (..),
    evaluate,
    runMetrics,
    printReport,
  )
where

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
