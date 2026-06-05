{-# LANGUAGE TypeApplications #-}

-- | Data layer (E) — the LOADER for the MNIST example: reads the IDX files and
--   forms the 'MnistDataset' format (from "MnistAddition.E_Data.Signature"),
--   prepared independently of any training or loss. Image pairs (x, y), the
--   observed sums [B] (raw indices, encoded by the D layer) used by the axiom, and the
--   per-pair sums/labels used to score accuracy. Only sums are "observed"; digit labels build the sums
--   and score afterwards. The raw IDX files sit beside this module in @E_Data/@
--   (see @Examples/MnistAddition/E_Data/get-mnist.sh@).
module MnistAddition.E_Data.Loader
  ( loadMnistDataset,
    loadData,
    batches,
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
  return
    MnistDataset
      { -- the batch carries the RAW observed sums ([B] tensor of indices); the D interpretation
        -- ('encodeObs') one-hots + encodes them into the LogVec delta (the @encode@ of the obs).
        trainBatch = (trX, trY, Torch.asTensor (map fromIntegral trS :: [Float])),
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

-- | Mini-batches of the training pairs (batch 64), re-SHUFFLED each epoch by a pure
--   per-epoch permutation @i \mapsto (a_e i + c_e) \bmod n@ (with @a_e@ coprime to @n@),
--   gathered via 'Torch.indexSelect''. Good SGD hygiene; deterministic in @epoch@ (no RNG).
batches :: Int -> MnistDataset -> [(Torch.Tensor, Torch.Tensor, Torch.Tensor)]
batches epoch ds =
  let (xs, ys, hs) = trainBatch ds
      n = head (Torch.shape xs)
      mults = [997, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117]
      a = mults !! (epoch `mod` length mults) -- coprime to n (prime > 5), so the map is a bijection
      perm = [(a * i + 137 * epoch) `mod` n | i <- [0 .. n - 1]]
      gather t = Torch.indexSelect' 0 perm t
      (xs', ys', hs') = (gather xs, gather ys, gather hs)
      bs = 64
      slice t s = Torch.sliceDim 0 s (min (s + bs) n) 1 t
   in [(slice xs' s, slice ys' s, slice hs' s) | s <- [0, bs .. n - 1]]
