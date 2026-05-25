{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the Binary example: the standard
--   classification metrics (accuracy / F1 / precision / confidence) as a labeled
--   'Report', predicting with @classifierA \@MeasU@ and scoring against @labelA \@MeasU@.
module Binary.G_Statistical.Interpretation (report, binaryReport) where

import A_Categorical.CategoricalInterpretation (MeasU)
import Binary.C_Domain.Interpretation ()
import Binary.C_Domain.Signature (BinaryKlRel (..), BinaryRel (..), BinarySorts (..))
import Binary.E_Data.Signature (BinaryDataset (..))
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import G_Statistical.Report (Report, evaluate, runMetrics)
import C_Domain.Models.Interpretations.MLP (MLPSpace)
import qualified Torch

binaryReport :: MLPSpace -> BinaryDataset -> Report
binaryReport theta ds =
  let toPoints t = map (\[x1, x2] -> (x1, x2)) (Torch.asValue t :: [[Float]]) :: [Point MeasU]
      predict pt = distPTrue (classifierA @MeasU theta pt)
      label = labelA @MeasU
      trainPairs = evaluate predict label (toPoints (trainData ds))
      testPairs = evaluate predict label (toPoints (testData ds))
   in runMetrics trainPairs testPairs

-- | G-layer manifest piece for the Example.
report :: MLPSpace -> BinaryDataset -> IO Report
report theta ds = return (binaryReport theta ds)
