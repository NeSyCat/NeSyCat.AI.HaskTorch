{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The Binary classification experiment as a single 'Domain' instance: it
--   wires the binary interpretations (gamma: classifierA/labelA), its objective
--   (epsilon: lossForBinary) and its metrics (zeta: the standard report). That
--   is all an example contributes; 'runDomain' does the rest.
module Examples.Binary (Binary) where

import A_Categorical.CategoricalInterpretation (MeasU)
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import C_Domain.Examples.Binary.Interpretation ()
import C_Domain.Examples.Binary.Signature (BinaryKlRel (..), BinaryRel (..), BinarySorts (..))
import C_Domain.Models.MLP (ParamsMLP, ParamsMLPSpec (..), binarySpecReal)
import Example (Example (..))
import E_Inferential.Examples.Binary.Train (BinaryDataset (..), generateBinaryDataset, lossForBinary)
import F_Statistical.Report (evaluate, runMetrics)
import qualified Torch

-- | Tag type selecting the binary experiment.
data Binary

instance Example Binary where
  type Params Binary = ParamsMLP
  type Spec Binary = ParamsMLPSpec
  type Data Binary = BinaryDataset

  name = "Binary Classification (classifierA @MeasU)"
  spec = binarySpecReal
  trainConfig = (1000, 0.001)
  loadData = generateBinaryDataset
  objective ds theta = lossForBinary 1.75 1.0 ds theta -- beta=1.75, lambda=1.0 (pure axiom)
  report theta ds =
    let toPoints t = map (\[x1, x2] -> (x1, x2)) (Torch.asValue t :: [[Float]]) :: [Point MeasU]
        predict pt = distPTrue (classifierA @MeasU theta pt)
        label = labelA @MeasU
        trainPairs = evaluate predict label (toPoints (trainData ds))
        testPairs = evaluate predict label (toPoints (testData ds))
     in return (runMetrics trainPairs testPairs)
