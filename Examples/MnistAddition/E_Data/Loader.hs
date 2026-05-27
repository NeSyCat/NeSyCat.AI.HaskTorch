{-# LANGUAGE TypeApplications #-}

-- | Data layer (E) — the LOADER for the MNIST example: reads the IDX files and
--   forms the 'MnistDataset' format (from "MnistAddition.E_Data.Signature"),
--   prepared independently of any training or loss. Image pairs (x, y), the
--   one-hot observed sums [B,19] used by the axiom, and the per-pair sums/labels
--   used to score accuracy. Only sums are "observed"; digit labels build the sums
--   and score afterwards. The raw IDX files sit beside this module in @E_Data/@
--   (see @Examples/MnistAddition/E_Data/get-mnist.sh@).
module MnistAddition.E_Data.Loader
  ( loadMnistDataset,
    loadData,
  )
where

import Control.Monad (unless)
import MnistAddition.E_Data.Signature (Dataset, MnistDataset (..))
import System.Directory (doesFileExist)
import System.Exit (die)
import qualified Torch
import qualified Torch.Typed.Vision as V
import qualified Torch.Vision as TV

-- | Directory holding the four gzipped IDX files (see @Examples/MnistAddition/E_Data/get-mnist.sh@).
mnistDir :: String
mnistDir = "Examples/MnistAddition/E_Data"

-- | Read MNIST, form @nTr@/@nTe@ disjoint image pairs (x = even index, y = odd),
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
    die ("MNIST data not found in " ++ mnistDir ++ "/. Run:  Examples/MnistAddition/E_Data/get-mnist.sh")
  (trainMD, testMD) <- V.initMnist mnistDir
  let nTr = 3000
      nTe = 1000
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

-- | E-layer manifest piece for the Example: how to obtain the data.
loadData :: IO Dataset
loadData = loadMnistDataset
