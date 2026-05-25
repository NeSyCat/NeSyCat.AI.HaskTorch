{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The MNIST single-digit-addition example: the full A–G stack assembled. A/B
--   reuse the library; C domain, D grammatical, E inference (categorical NLL),
--   F statistics and G data are defined in this folder. 'runExample' does the rest.
module Examples.MnistAddition.Example (MnistAdd) where

import Examples.MnistAddition.A_Categorical ()
import Examples.MnistAddition.B_Logical ()
import Examples.MnistAddition.C_Domain.Interpretation ()
import Examples.MnistAddition.D_Grammatical.IntpTens (mnistSumLogits)
import Examples.MnistAddition.E_Inferential (mnistKnowLoss)
import Examples.MnistAddition.F_Statistical (mnistReport)
import Examples.MnistAddition.G_Data (MnistDataset (..), loadMnistDataset)
import Lib.Example (Example (..))
import Lib.Models.MnistCNN (ParamsCNN, ParamsCNNSpec (..))

data MnistAdd

instance Example MnistAdd where
  type Params MnistAdd = ParamsCNN
  type Spec MnistAdd = ParamsCNNSpec
  type Data MnistAdd = MnistDataset

  name = "MNIST single-digit addition (axiom-only; digits learned from sums)"
  spec = ParamsCNNSpec
  trainConfig = (200, 0.001)
  loadData = loadMnistDataset
  -- E (inference) categorical NLL of the D (grammatical) GeomU sum-logits, target = observed sums.
  objective ds theta =
    let (xB, yB, oneHotSums) = trainBatch ds
     in mnistKnowLoss (mnistSumLogits theta (xB, yB)) oneHotSums
  report theta ds = return (mnistReport theta ds)
