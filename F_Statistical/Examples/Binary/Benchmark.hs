{-# LANGUAGE TypeApplications #-}

-- | Binary classification benchmark (the runnable example).
--
--   1. Trains theta* via E_Inferential (epsilon level).
--   2. Scores classifierA @MeasU (gamma) with the zeta-level metrics.
--
--   All reporting is reused from F_Statistical.Report, so this driver only
--   chooses the universe, the model application, and the data.
--
--   Run:  cabal run binary-benchmark
module Main where

import A_Categorical.CategoricalInterpretation (MeasU)
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import C_Domain.Examples.Binary.Interpretation ()
import C_Domain.Examples.Binary.Signature (BinaryKlRel (..), BinaryRel (..), BinarySorts (..))
import E_Inferential.Examples.Binary.Train (BinaryDataset (..), generateBinaryDataset, trainBinary)
import F_Statistical.Report (evaluate, printReport, runMetrics)
import qualified Torch

main :: IO ()
main = do
  ds <- generateBinaryDataset
  thetaStar <- trainBinary 1000 0.001 1.0 1.75 ds

  let toPoints t = map (\[x1, x2] -> (x1, x2)) (Torch.asValue t :: [[Float]]) :: [Point MeasU]
      predict pt = distPTrue (classifierA @MeasU thetaStar pt)
      label = labelA @MeasU
      trainPairs = evaluate predict label (toPoints (trainData ds))
      testPairs = evaluate predict label (toPoints (testData ds))

  printReport "Binary Benchmark (classifierA @MeasU)" (runMetrics trainPairs testPairs)
