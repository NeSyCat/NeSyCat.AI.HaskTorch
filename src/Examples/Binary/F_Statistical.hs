{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer for the Binary example: the standard classification metrics
--   (accuracy / F1 / precision / confidence) as a labeled 'Report', predicting
--   with @classifierA \@MeasU@ and scoring against @labelA \@MeasU@.
module Examples.Binary.F_Statistical (binaryReport) where

import Examples.Binary.A_Categorical (MeasU)
import Examples.Binary.C_Domain.Interpretation ()
import Examples.Binary.C_Domain.Signature (BinaryKlRel (..), BinaryRel (..), BinarySorts (..))
import Examples.Binary.G_Data (BinaryDataset (..))
import Lib.A_Categorical.Category.Monads.DistExpect (distPTrue)
import Lib.F_Statistical.Report (Report, evaluate, runMetrics)
import Lib.Models.MLP (ParamsMLP)
import qualified Torch

binaryReport :: ParamsMLP -> BinaryDataset -> Report
binaryReport theta ds =
  let toPoints t = map (\[x1, x2] -> (x1, x2)) (Torch.asValue t :: [[Float]]) :: [Point MeasU]
      predict pt = distPTrue (classifierA @MeasU theta pt)
      label = labelA @MeasU
      trainPairs = evaluate predict label (toPoints (trainData ds))
      testPairs = evaluate predict label (toPoints (testData ds))
   in runMetrics trainPairs testPairs
