{-# LANGUAGE TypeApplications #-}

-- | Data layer (E) — the LOADER for the MNIST example: reads the IDX files and
--   forms the 'MnistDataset' format (from "MnistAddition.E_Data.Signature"),
--   prepared independently of any training or loss. Image pairs (x, y), the
--   observed sums as @eta n@ (a LogTens leaf over [0..18], the distributional format the axiom binds),
--   and the per-pair sums/labels used to score accuracy. Only sums are "observed"; digit labels build the sums
--   and score afterwards. The raw IDX files sit beside this module in @E_Data/@
--   (see @Examples/MnistAddition/E_Data/get-mnist.sh@).
module MnistAddition.E_Data.Loader
  ( loadMnistDataset,
    loadData,
    batches,
  )
where

import A_Categorical.Monads.Bridge (encode)
import A_Categorical.Monads.LogTens (LogTens)
import A_Categorical.Monads.LogTensExpect (mapLeafWeights)
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
      -- the observed sums, already in the right (distributional) format: @eta n@ as a LogTens leaf
      -- over [0..18] (one-hot, encoded). Built ONCE here; mini-batched by 'batches'. The D layer
      -- then just binds it (@s <- n@) -- no lifting in the interpretation.
      oneHot ss = Torch.asTensor [[if s == k then 1.0 else 0.0 :: Float | k <- [0 .. 18]] | s <- ss]
      obsAll = encode [0 .. 18] (oneHot trS)
  return
    MnistDataset
      { trainBatch = (trX, trY, obsAll),
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
batches :: Int -> MnistDataset -> [(LogTens Torch.Tensor, LogTens Torch.Tensor, LogTens Int)]
batches epoch ds =
  let (xs, ys, obs) = trainBatch ds
      total = head (Torch.shape xs)
      mults = [997, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117]
      a = mults !! (epoch `mod` length mults) -- coprime to total (prime > 5), so the map is a bijection
      perm = [(a * i + 137 * epoch) `mod` total | i <- [0 .. total - 1]]
      gather t = Torch.indexSelect' 0 perm t -- shuffle the images (tensors) ...
      obsG = mapLeafWeights (Torch.indexSelect' 0 perm) obs -- ... and the observation leaf, in lockstep
      (xs', ys') = (gather xs, gather ys)
      batchSize = 32
      slice t start = Torch.sliceDim 0 start (min (start + batchSize) total) 1 t
   in [(pure (slice xs' start), pure (slice ys' start), mapLeafWeights (\lw -> slice lw start) obsG) | start <- [0, batchSize .. total - 1]]
      -- images are encoded at load: each batch carries them as @eta x = pure x :: LogTens Image@ (a certain leaf)
