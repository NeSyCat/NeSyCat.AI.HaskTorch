{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Statistics layer (G) — INTERPRETATION for the Binary example: the standard
--   classification metrics (accuracy / F1 / precision / confidence) as a labeled
--   'Report', predicting with @classifierA \@Dist@ and scoring against @labelA \@Dist@.
module Binary.G_Statistical.Interpretation (report, binaryReport) where

import A_Categorical.Category.Monads.Dist (Dist)
import Binary.C_Domain.Interpretation ()
import Binary.C_Domain.Signature (BinaryKlRel (..), BinaryRel (..), Point)
import Binary.E_Data.Signature (BinaryDataset (..))
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import G_Statistical.Report (Report, evaluate, runMetrics)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

binaryReport :: Weights -> BinaryDataset -> Report
binaryReport theta ds =
  let toPoints t = map Torch.asTensor (Torch.asValue t :: [[Float]]) :: [Point] -- each row -> a [2] tensor
      predict pt = distPTrue (classifierA @Dist theta pt)
      label pt = distPTrue (labelA @Dist pt) > 0.5  -- labelA is now a (certain) Dist Bool
      trainPairs = evaluate predict label (toPoints (trainData ds))
      testPairs = evaluate predict label (toPoints (testData ds))
   in runMetrics trainPairs testPairs

-- | G-layer manifest piece for the Example.
report :: Weights -> BinaryDataset -> IO Report
report theta ds = return (binaryReport theta ds)
