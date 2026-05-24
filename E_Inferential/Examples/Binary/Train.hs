{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Binary classification training (epsilon level, binary example).
--   Minimizes the inferential objective  J = (1-lambda)*J_data + lambda*J_know
--   over the MLP parameters, returning theta*. Also generates the dataset,
--   whose ground-truth labels are produced by the domain's @labelA@ (no
--   duplicated circle-in-square logic).
module E_Inferential.Examples.Binary.Train
  ( trainBinary,
    generateBinaryDataset,
    BinaryDataset (..),
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import C_Domain.Examples.Binary.Interpretation ()
import C_Domain.Examples.Binary.Signature (BinaryRel (..))
import C_Domain.Models.MLP (ParamsMLP, binarySpecReal, hThetaReal)
import D_Grammatical.Examples.Binary.IntpTens (binaryAxiomTens)
import E_Inferential.InferenceInterpretation ()
import E_Inferential.InferenceSignature (InferenceSignature (..))

import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Text.Printf (printf)
import Torch (Parameterized (..), Randomizable (..))
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))
import Torch.NN ()
import Torch.Optim (mkAdam, runStep)
import Torch.Tensor (toDevice)

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
  let logits = labelA @GeomU dataset -- circle-in-square, via the domain's labelA
      labels = Torch.toType Torch.Float (logits `Torch.gt` Torch.asTensor (0.0 :: Float))
  return
    BinaryDataset
      { trainData = Torch.sliceDim 0 0 50 1 dataset,
        trainLabels = Torch.reshape [50, 1] (Torch.sliceDim 0 0 50 1 labels),
        testData = Torch.sliceDim 0 0 50 1 (Torch.sliceDim 0 50 100 1 dataset),
        testLabels = Torch.reshape [50, 1] (Torch.sliceDim 0 0 50 1 (Torch.sliceDim 0 50 100 1 labels))
      }

-- | Train via the inferential objective. Returns optimized theta*.
--   @verbose@ controls per-epoch logging (off during multi-run averaging).
trainBinary ::
  Bool ->   -- verbose (print the loss curve)
  Int ->    -- epochs
  Float ->  -- learning rate
  Float ->  -- lambda (0=pure data, 1=pure axiom)
  Float ->  -- beta (LogSumExp sharpness)
  BinaryDataset ->
  IO ParamsMLP
trainBinary verbose numEpochs learningRate lambda betaFixed ds = do
  initModel <- toDevice (Device CPU 0) <$> sample binarySpecReal
  let initOpt = mkAdam 0 0.9 0.999 (flattenParameters initModel)
      td = trainData ds
      tl = trainLabels ds
      say s = if verbose then putStrLn s else return ()

  say $
    printf "[Training] %d epochs, beta=%.2f, lr=%.4f, lambda=%.2f"
      numEpochs betaFixed learningRate lambda

  let !_ = td `seq` tl `seq` ()

  startTime <- getCurrentTime
  let betaT = Torch.asTensor betaFixed
      lrTens = Torch.toDevice (Device CPU 0) (Torch.asTensor learningRate)
      nTens = Torch.toDevice (Device CPU 0) (Torch.asTensor (fromIntegral (head (Torch.shape td)) :: Float))
      zeroTens = Torch.asTensor (0.0 :: Float)
      lambdaTens = Torch.asTensor lambda

  (paramMLPOpti, _) <- foldLoop (initModel, initOpt) [1 .. numEpochs] $ \(model, opt) epoch -> do
    let dataLoss =
          if lambda == 1.0
            then zeroTens
            else
              let preds = Torch.sigmoid (hThetaReal model td)
               in Torch.sumAll (lossData preds tl) `Torch.div` nTens
    let knowLoss =
          if lambda == 0.0
            then zeroTens
            else lossKnow (binaryAxiomTens betaT td model)
    let totalLoss = lossComb dataLoss knowLoss lambdaTens
    (newModel, newOpt) <- runStep model opt totalLoss lrTens
    if verbose && (epoch `mod` 100 == 0 || epoch == numEpochs || epoch == 1)
      then do
        epochEnd <- getCurrentTime
        let totalVal = Torch.asValue totalLoss :: Float
            diffMs = (realToFrac (diffUTCTime epochEnd startTime) :: Double) * 1000
        putStrLn $ printf "[Epoch %3d/%d] J=%7.5f | %.2fms" epoch numEpochs totalVal diffMs
      else return ()
    return (newModel, newOpt)

  totalEnd <- getCurrentTime
  let totalDiff = realToFrac (diffUTCTime totalEnd startTime) :: Double
  say $ printf "[Training complete] %.2fs" totalDiff

  return paramMLPOpti

foldLoop :: a -> [b] -> (a -> b -> IO a) -> IO a
foldLoop acc [] _ = return acc
foldLoop acc (x : xs) f = f acc x >>= \a -> foldLoop a xs f
