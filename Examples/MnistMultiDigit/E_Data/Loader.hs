{-# LANGUAGE TypeApplications #-}

-- | Data layer (E) — the LOADER for the MNIST multi-digit example: reads the IDX files and forms
--   the 'MultiDataset'. Image QUADRUPLES (x1,x2 = a two-digit number A; y1,y2 = a two-digit number
--   B), the observed sums @N = number A + number B@ in [0..198] as @eta n@ (a @LogTens Int@ leaf over
--   [0..198], the distributional format the axiom binds), and the per-digit labels used to score
--   accuracy. Only sums are "observed". Reuses the IDX files shipped with the single-digit example
--   (@Examples/MnistAddition/E_Data/@).
module MnistMultiDigit.E_Data.Loader
  ( loadMultiDataset,
    loadData,
    batches,
  )
where

import A_Categorical.Monads.Bridge (encode)
import A_Categorical.Monads.LogTens (LogTens)
import A_Categorical.Monads.LogTensExpect (mapLeafWeights)
import Control.Monad (unless)
import MnistMultiDigit.E_Data.Signature (Dataset, MultiDataset (..))
import System.Directory (doesFileExist)
import System.Exit (die)
import qualified Torch
import qualified Torch.Typed.Vision as V
import qualified Torch.Vision as TV

-- | The single-digit example's IDX files are reused (the same MNIST data).
mnistDir :: String
mnistDir = "Examples/MnistAddition/E_Data"

-- | Read MNIST, form @nTr@/@nTe@ disjoint image quadruples (x1,x2,y1,y2 at indices 4i..4i+3),
--   normalize to [0,1], and record the observed sums @N = (10*x1+x2) + (10*y1+y2)@.
loadMultiDataset :: IO MultiDataset
loadMultiDataset = do
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
  let nTr = 1500 -- LTN's small-data multi-digit setting (Table 2 "1500" column)
      nTe = 2500 -- DeepProbLog's multi-digit test size (all 10000 t10k images / 4 = 2500 quadruples)
      buildQuads md nQuads =
        let idxAt off = [4 * i + off | i <- [0 .. nQuads - 1]]
            (x1i, x2i, y1i, y2i) = (idxAt 0, idxAt 1, idxAt 2, idxAt 3)
            imgs idxs =
              Torch.reshape [nQuads, 1, 28, 28] $
                TV.getImages' nQuads 784 md idxs `Torch.div` Torch.asTensor (255.0 :: Float)
            lab = map (V.getLabel md)
            (l1, l2, l3, l4) = (lab x1i, lab x2i, lab y1i, lab y2i)
            twoDigit = zipWith (\hi lo -> 10 * hi + lo)
            sums = zipWith (+) (twoDigit l1 l2) (twoDigit l3 l4)
         in (imgs x1i, imgs x2i, imgs y1i, imgs y2i, l1, l2, l3, l4, sums)
      (trX1, trX2, trY1, trY2, _, _, _, _, trS) = buildQuads trainMD nTr
      (teX1, teX2, teY1, teY2, te1, te2, te3, te4, teS) = buildQuads testMD nTe
      -- the observed sums, already in the right (distributional) format: @eta n@ as a LogTens leaf
      -- over [0..198] (one-hot, encoded). Built ONCE here; mini-batched by 'batches'.
      oneHot ss = Torch.asTensor [[if s == k then 1.0 else 0.0 :: Float | k <- [0 .. 198]] | s <- ss]
      obsAll = encode [0 .. 198] (oneHot trS)
  return
    MultiDataset
      { trainBatch = (trX1, trX2, trY1, trY2, obsAll),
        trainX1 = trX1,
        trainX2 = trX2,
        trainY1 = trY1,
        trainY2 = trY2,
        trainSums = trS,
        testX1 = teX1,
        testX2 = teX2,
        testY1 = teY1,
        testY2 = teY2,
        testSums = teS,
        testLabs = (te1, te2, te3, te4)
      }

-- | E-layer manifest piece for the Example: how to obtain the data.
loadData :: IO Dataset
loadData = loadMultiDataset

-- | Mini-batches of the training quadruples (batch 64), re-SHUFFLED each epoch by a pure per-epoch
--   permutation (a bijection; deterministic in @epoch@, no RNG). The four image tensors and the
--   observation leaf are gathered/sliced in lockstep.
batches :: Int -> MultiDataset -> [(Torch.Tensor, Torch.Tensor, Torch.Tensor, Torch.Tensor, LogTens Int)]
batches epoch ds =
  let (xs1, xs2, ys1, ys2, obs) = trainBatch ds
      n = head (Torch.shape xs1)
      mults = [997, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117]
      a = mults !! (epoch `mod` length mults) -- coprime to n, so the map is a bijection
      perm = [(a * i + 137 * epoch) `mod` n | i <- [0 .. n - 1]]
      gather t = Torch.indexSelect' 0 perm t
      obsG = mapLeafWeights (Torch.indexSelect' 0 perm) obs
      (g1, g2, g3, g4) = (gather xs1, gather xs2, gather ys1, gather ys2)
      bs = 32
      slice t s = Torch.sliceDim 0 s (min (s + bs) n) 1 t
   in [ (slice g1 s, slice g2 s, slice g3 s, slice g4 s, mapLeafWeights (\lw -> slice lw s) obsG)
        | s <- [0, bs .. n - 1]
      ]
