{-# LANGUAGE TypeApplications #-}

-- | Binary classification example (epsilon level): the dataset and the
--   inferential objective J(theta) = (1-lambda)*J_data + lambda*J_know.
--   The optimization loop is the generic 'E_Inferential.Train.train'; the
--   wiring into the generic runner lives in "Examples.Binary".
module E_Inferential.Examples.Binary.Train
  ( lossForBinary,
    generateBinaryDataset,
    BinaryDataset (..),
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import C_Domain.Examples.Binary.Interpretation ()
import C_Domain.Examples.Binary.Signature (BinaryRel (..))
import C_Domain.Models.MLP (ParamsMLP, hThetaReal)
import D_Grammatical.Examples.Binary.IntpTens (binaryAxiomTens)
import E_Inferential.InferenceInterpretation ()
import E_Inferential.InferenceSignature (InferenceSignature (..))
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))

-- | A binary classification dataset (circle-in-square). 50 train / 50 test.
data BinaryDataset = BinaryDataset
  { trainData :: Torch.Tensor,
    trainLabels :: Torch.Tensor,
    testData :: Torch.Tensor,
    testLabels :: Torch.Tensor
  }

-- | Generate 100 random points in [0,1]^2; labels come from @labelA \@GeomU@
--   (batched), so the ground-truth concept lives only in the domain interpretation.
generateBinaryDataset :: IO BinaryDataset
generateBinaryDataset = do
  dataset <- Torch.toDevice (Device CPU 0) <$> Torch.randIO' [100, 2]
  let logits = labelA @GeomU dataset
      labels = Torch.toType Torch.Float (logits `Torch.gt` Torch.asTensor (0.0 :: Float))
  return
    BinaryDataset
      { trainData = Torch.sliceDim 0 0 50 1 dataset,
        trainLabels = Torch.reshape [50, 1] (Torch.sliceDim 0 0 50 1 labels),
        testData = Torch.sliceDim 0 0 50 1 (Torch.sliceDim 0 50 100 1 dataset),
        testLabels = Torch.reshape [50, 1] (Torch.sliceDim 0 0 50 1 (Torch.sliceDim 0 50 100 1 labels))
      }

-- | The binary inferential objective, closing over the dataset: this is the
--   @params -> Tensor@ the generic trainer minimizes.
lossForBinary :: Float -> Float -> BinaryDataset -> ParamsMLP -> Torch.Tensor
lossForBinary betaFixed lambda ds model =
  let td = trainData ds
      tl = trainLabels ds
      betaT = Torch.asTensor betaFixed
      nTens = Torch.toDevice (Device CPU 0) (Torch.asTensor (fromIntegral (head (Torch.shape td)) :: Float))
      zeroTens = Torch.asTensor (0.0 :: Float)
      lambdaTens = Torch.asTensor lambda
      dataLoss =
        if lambda == 1.0
          then zeroTens
          else
            let preds = Torch.sigmoid (hThetaReal model td)
             in Torch.sumAll (lossData preds tl) `Torch.div` nTens
      knowLoss =
        if lambda == 0.0
          then zeroTens
          else lossKnow (binaryAxiomTens betaT td model)
   in lossComb dataLoss knowLoss lambdaTens
