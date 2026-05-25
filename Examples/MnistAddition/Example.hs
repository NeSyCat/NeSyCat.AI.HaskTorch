{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The MNIST single-digit-addition example: the full A–G stack assembled. A/B
--   reuse the library; C domain, D grammatical, E inference (categorical NLL),
--   F statistics and G data are defined in this folder. 'runExample' does the rest.
module MnistAddition.Example (MnistAddition) where

import MnistAddition.A_Categorical ()
import MnistAddition.B_Logical ()
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.D_Grammatical.InterpretationTens (mnistSumLogits)
import MnistAddition.E_Inferential (mnistKnowLoss)
import MnistAddition.F_Statistical (mnistReport)
import MnistAddition.G_Data.Loader (MnistDataset (..), loadMnistDataset)
import Example (Example (..))
import C_Domain.Models.MnistCNN (ParamsCNN, ParamsCNNSpec (..))

data MnistAddition

instance Example MnistAddition where
  type Params MnistAddition = ParamsCNN
  type Spec MnistAddition = ParamsCNNSpec
  type Data MnistAddition = MnistDataset

  name = "MNIST single-digit addition (axiom-only; digits learned from sums)"
  spec = ParamsCNNSpec
  trainConfig = (200, 0.001)
  loadData = loadMnistDataset
  -- E (inference) categorical NLL of the D (grammatical) GeomU sum-logits, target = observed sums.
  objective ds theta =
    let (xB, yB, oneHotSums) = trainBatch ds
     in mnistKnowLoss (mnistSumLogits theta (xB, yB)) oneHotSums
  report theta ds = return (mnistReport theta ds)
