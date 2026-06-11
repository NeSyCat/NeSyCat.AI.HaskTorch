-- | Data layer (E) -- the LOADER for WAP: reads the committed reference data
--   (@Examples/WAP/E_Data/{train,dev,test}.txt@ + @vocab_746.txt@, fetched verbatim from the
--   Apache-2.0 DeepProbLog repo) and tokenizes 1:1 with the reference @wap_network.py@:
--   whitespace split; every all-digit token becomes @\<NR\>@ (its value and position
--   recorded -- there are always exactly three); out-of-vocabulary words become @\<UNK\>@;
--   vocab ids are line numbers in @vocab_746.txt@.
module WAP.E_Data.Loader
  ( loadWapDataset,
    loadData,
    batches,
  )
where

import A_Categorical.Monads.Bridge (encode)
import A_Categorical.Monads.LogVec (LogVec)
import Control.Monad (unless)
import Data.Char (isDigit)
import Data.List (nub)
import qualified Data.Map.Strict as M
import System.Directory (doesFileExist)
import System.Exit (die)
import WAP.C_Domain.Signature (Answer, Numbers, Problem)
import WAP.E_Data.Signature (Dataset, WapDataset (..), WapItem)
import qualified Torch

-- | Directory holding the committed reference data files.
wapDir :: String
wapDir = "Examples/WAP/E_Data"

-- | Read the vocab (id = line number) and the three splits.
loadWapDataset :: IO WapDataset
loadWapDataset = do
  let files = ["train.txt", "dev.txt", "test.txt", "vocab_746.txt"]
  present <- mapM (\f -> doesFileExist (wapDir ++ "/" ++ f)) files
  unless (and present) $
    die ("WAP data not found in " ++ wapDir ++ "/ (train/dev/test/vocab_746).")
  vocabLines <- lines <$> readFile (wapDir ++ "/vocab_746.txt")
  let vocab = M.fromList (zip vocabLines [0 ..])
      parseSplit f = do
        ls <- lines <$> readFile (wapDir ++ "/" ++ f)
        return (map (parseItem vocab) (filter (not . null) ls))
  tr <- parseSplit "train.txt"
  dv <- parseSplit "dev.txt"
  te <- parseSplit "test.txt"
  return WapDataset {trainItems = tr, devItems = dv, testItems = te}

-- | One line: @answer\<TAB\>problem text@. Tokenization follows the reference exactly.
parseItem :: M.Map String Int -> String -> WapItem
parseItem vocab line =
  let (ansS, rest) = break (== '\t') line
      text = drop 1 rest
      y = round (read ansS :: Double) :: Answer
      toks = words text
      step (ids, nums, poss) (i, w)
        | not (null w) && all isDigit w =
            (ids ++ [vocab M.! "<NR>"], nums ++ [read w], poss ++ [i])
        | otherwise =
            (ids ++ [M.findWithDefault (vocab M.! "<UNK>") w vocab], nums, poss)
      (tokIds, numbers, positions) = foldl step ([], [], []) (zip [0 ..] toks)
      ns = case numbers of
        [a, b, c] -> (a, b, c) :: Numbers
        _ -> error ("WAP item without exactly 3 numbers: " ++ text)
   in ((tokIds, positions), ns, y)

-- | E-layer manifest piece for the Example: how to obtain the data.
loadData :: IO Dataset
loadData = loadWapDataset

-- | Mini-batches of the training problems (batch 10, the reference DataLoader size),
--   re-SHUFFLED each epoch by a pure per-epoch permutation @i \mapsto (a_e i + c_e) \bmod n@
--   (with @a_e@ coprime to @n@) -- the same deterministic SGD hygiene as the MNIST examples.
--   Each batch carries its observation as @\eta (ns, y)@: a one-hot @LogVec@ leaf over the
--   batch's DISTINCT (numbers, answer) pairs (the @encode@ of the certain observation --
--   the distributional format the axiom binds, built HERE so the D interpretation is a pure
--   pass-through, exactly the MNIST pattern).
batches :: Int -> WapDataset -> [([Problem], LogVec (Numbers, Answer))]
batches epoch ds =
  let items = trainItems ds
      n = length items
      mults = [997, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117]
      a = mults !! (epoch `mod` length mults) -- prime > 5, coprime to n
      perm = [(a * i + 137 * epoch) `mod` n | i <- [0 .. n - 1]]
      shuffled = [items !! p | p <- perm]
      bs = 10
      chunk [] = []
      chunk xs = take bs xs : chunk (drop bs xs)
      obsLeaf ch =
        let pairs = [(ns, y) | (_, ns, y) <- ch]
            support = nub pairs
            oneHot = [[if pr == s then 1.0 else 0.0 :: Float | s <- support] | pr <- pairs]
         in encode support (Torch.asTensor oneHot)
   in [(map (\(p, _, _) -> p) ch, obsLeaf ch) | ch <- chunk shuffled]
