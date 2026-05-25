{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The Binary classification example: the full A–G stack assembled into one
--   'Example' instance. Each layer is either REUSED from the library (A category,
--   B logic, E inference) or DEFINED in this folder (C domain, D grammatical,
--   F statistics, G data). 'runExample' trains and benchmarks it.
module Binary.Example (Binary) where

import Binary.A_Categorical ()
import Binary.B_Logical ()
import Binary.C_Domain.Interpretation ()
import Binary.D_Grammatical.InterpretationTens (binaryAxiomTens)
import Binary.E_Inferential (lossKnow)
import Binary.F_Statistical (binaryReport)
import Binary.G_Data.Loader (BinaryDataset (..), generateBinaryDataset)
import Example (Example (..))
import C_Domain.Models.MLP (ParamsMLP, ParamsMLPSpec (..), binarySpecReal)
import qualified Torch

data Binary

instance Example Binary where
  type Params Binary = ParamsMLP
  type Spec Binary = ParamsMLPSpec
  type Data Binary = BinaryDataset

  name = "Binary Classification"
  spec = binarySpecReal
  trainConfig = (1000, 0.001)
  loadData = generateBinaryDataset
  -- E (inference) penalty of the D (grammatical) axiom on the data; A/B/C reused/defined.
  objective ds theta = lossKnow (binaryAxiomTens (Torch.asTensor (1.75 :: Float)) (trainData ds) theta)
  report theta ds = return (binaryReport theta ds)
