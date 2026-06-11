-- | Data layer (E) -- SIGNATURE for WAP: the fixed FORMAT of the data. Each item is one
--   word problem (tokenized text + the three numbers in text order) with its observed
--   numeric answer -- the answers and numbers are PLAIN host integers (conditioning data;
--   never monadic values). Splits follow the committed reference data: 300 train / 100 dev
--   / 200 test (Roy & Roth's Common Core set, via the Apache-2.0 DeepProbLog repo).
module WAP.E_Data.Signature
  ( WapDataset (..),
    Dataset,
    WapItem,
  )
where

import WAP.C_Domain.Signature (Answer, Numbers, Problem)

-- | One word problem: the tokenized text, its three numbers (text order), its answer.
type WapItem = (Problem, Numbers, Answer)

-- | The WAP dataset, in the reference splits.
data WapDataset = WapDataset
  { trainItems :: [WapItem],
    devItems :: [WapItem],
    testItems :: [WapItem]
  }

-- | The canonical dataset type the Example manifest refers to.
type Dataset = WapDataset
