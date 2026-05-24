{-# LANGUAGE TypeApplications #-}

-- | Binary classification benchmark (the runnable example).
--
--   Trains theta* (epsilon) then scores classifierA @MeasU (gamma) with the
--   zeta-level metrics, averaging natively over N runs.
--
--   Run:  cabal run binary-benchmark         -- 10 runs, averaged
--         cabal run binary-benchmark -- 1     -- single run, with loss curve
--         cabal run binary-benchmark -- 50    -- 50 runs, averaged
module Main where

import A_Categorical.CategoricalInterpretation (MeasU)
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import C_Domain.Examples.Binary.Interpretation ()
import C_Domain.Examples.Binary.Signature (BinaryKlRel (..), BinaryRel (..), BinarySorts (..))
import E_Inferential.Examples.Binary.Train (BinaryDataset (..), generateBinaryDataset, trainBinary)
import F_Statistical.Report (evaluate, runAverage, runMetrics)
import System.Environment (getArgs)
import qualified Torch

main :: IO ()
main = do
  args <- getArgs
  let n = case args of (x : _) -> read x; _ -> 10 :: Int

  runAverage "Binary Benchmark (classifierA @MeasU)" n $ do
    ds <- generateBinaryDataset
    thetaStar <- trainBinary (n == 1) 1000 0.001 1.0 1.75 ds
    let toPoints t = map (\[x1, x2] -> (x1, x2)) (Torch.asValue t :: [[Float]]) :: [Point MeasU]
        predict pt = distPTrue (classifierA @MeasU thetaStar pt)
        label = labelA @MeasU
        trainPairs = evaluate predict label (toPoints (trainData ds))
        testPairs = evaluate predict label (toPoints (testData ds))
    return (runMetrics trainPairs testPairs)
