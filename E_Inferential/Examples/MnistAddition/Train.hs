{-# LANGUAGE TypeApplications #-}

-- | MNIST single-digit-addition example (epsilon level): the dataset of image
--   pairs with their observed sums, and the inferential objective. Only sums are
--   supervised; the digit labels are used solely to build the sums (and to score
--   digit accuracy afterwards). The objective is @lossKnow@ (softplus) applied to
--   the GeomU axiom 'mnistAxiomTens' -- the same epsilon machinery as binary. The
--   optimization loop is the generic 'E_Inferential.Train.train'.
module E_Inferential.Examples.MnistAddition.Train
  ( MnistDataset (..),
    loadMnistDataset,
    lossForMnist,
    mnistBeta,
  )
where

import Control.Monad (unless)
import C_Domain.Models.MnistCNN (ParamsCNN)
import D_Grammatical.Examples.MnistAddition.IntpTens (mnistAxiomTens)
import E_Inferential.InferenceInterpretation ()
import E_Inferential.InferenceSignature (InferenceSignature (..))
import System.Directory (doesFileExist)
import System.Exit (die)
import qualified Torch
import qualified Torch.Typed.Vision as V
import qualified Torch.Vision as TV

-- | Directory holding the four gzipped IDX files (see @scripts/get-mnist.sh@).
mnistDir :: String
mnistDir = "data/mnist"

-- | Logic smoothing parameter beta for the GeomU quantifier/connectives.
mnistBeta :: Float
mnistBeta = 2.0

-- | A dataset of image pairs. @trainBatch@ is the differentiable training input
--   (images x, images y, one-hot observed sums [B,19]); the @*Img@/@*Lab@/@*Sums@
--   fields back the (argmax) accuracy report.
data MnistDataset = MnistDataset
  { trainBatch :: (Torch.Tensor, Torch.Tensor, Torch.Tensor),
    trainXImg :: Torch.Tensor,
    trainYImg :: Torch.Tensor,
    trainSums :: [Int],
    testXImg :: Torch.Tensor,
    testYImg :: Torch.Tensor,
    testSums :: [Int],
    testXLab :: [Int],
    testYLab :: [Int]
  }

-- | Load MNIST, form @nTr@/@nTe@ disjoint image pairs (x = even index, y = odd),
--   normalize to [0,1], and record the observed sums. Errors with guidance if the
--   IDX files are missing.
loadMnistDataset :: IO MnistDataset
loadMnistDataset = do
  let files =
        [ "train-images-idx3-ubyte.gz",
          "train-labels-idx1-ubyte.gz",
          "t10k-images-idx3-ubyte.gz",
          "t10k-labels-idx1-ubyte.gz"
        ]
  present <- mapM (\f -> doesFileExist (mnistDir ++ "/" ++ f)) files
  unless (and present) $
    die ("MNIST data not found in " ++ mnistDir ++ "/. Run:  scripts/get-mnist.sh")
  (trainMD, testMD) <- V.initMnist mnistDir
  let nTr = 1000
      nTe = 500
      buildPairs md nPairs =
        let xIdx = [2 * i | i <- [0 .. nPairs - 1]]
            yIdx = [2 * i + 1 | i <- [0 .. nPairs - 1]]
            imgs idxs =
              Torch.reshape [nPairs, 1, 28, 28] $
                TV.getImages' nPairs 784 md idxs `Torch.div` Torch.asTensor (255.0 :: Float)
            xLab = map (V.getLabel md) xIdx
            yLab = map (V.getLabel md) yIdx
         in (imgs xIdx, imgs yIdx, xLab, yLab, zipWith (+) xLab yLab)
      (trX, trY, _, _, trS) = buildPairs trainMD nTr
      (teX, teY, teXL, teYL, teS) = buildPairs testMD nTe
      oneHot ss =
        Torch.asTensor [[if s == k then 1.0 else 0.0 :: Float | k <- [0 .. 18]] | s <- ss]
  return
    MnistDataset
      { trainBatch = (trX, trY, oneHot trS),
        trainXImg = trX,
        trainYImg = trY,
        trainSums = trS,
        testXImg = teX,
        testYImg = teY,
        testSums = teS,
        testXLab = teXL,
        testYLab = teYL
      }

-- | The MNIST inferential objective: @lossKnow@ (softplus) of the GeomU axiom.
--   This is the @params -> Tensor@ the generic trainer minimizes (pure axiom).
lossForMnist :: MnistDataset -> ParamsCNN -> Torch.Tensor
lossForMnist ds theta =
  lossKnow (mnistAxiomTens (Torch.asTensor mnistBeta) (trainBatch ds) theta)
