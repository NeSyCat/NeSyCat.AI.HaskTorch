{-# LANGUAGE TypeApplications #-}

-- | Statistics layer (G) -- INTERPRETATION for WAP: answer accuracy on each split, under
--   the reference evaluation protocol (DeepProbLog's test networks run with k = 1): each
--   head decodes by ARGMAX independently, the decoded sketch is evaluated by the host
--   'evalSketch', and the answer must match exactly. Predictions are argmax over the
--   @LogVec@ logits (no softmax).
module WAP.G_Statistical.Interpretation (report) where

import A_Categorical.Monads.LogVec (LogVec)
import A_Categorical.Monads.LogVecExpect (logVecLeafTensor)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import G_Statistical.Report (Report (..))
import WAP.C_Domain.Interpretation (WapFun (..), WapKlFun (..))
import WAP.C_Domain.Signature (Answer, Numbers, Problem)
import WAP.E_Data.Signature (WapDataset (..), WapItem)
import qualified Torch

-- | Argmax decode of one head's leaf (the symbol's @LogVec@ logits, no softmax).
argmax1 :: LogVec a -> Int
argmax1 leaf =
  head (Torch.asValue (Torch.argmax (Torch.Dim 1) Torch.RemoveDim (logVecLeafTensor leaf)) :: [Int])

-- | The k=1 prediction: ONE trunk forward (the Fun symbol 'repF'), per-head argmax over
--   the Kleisli symbols' leaves, then the host sketch evaluation.
predictAnswer :: Weights -> Problem -> Numbers -> Maybe Answer
predictAnswer theta p ns =
  let r = repF theta [p]
      perm = argmax1 (permuteS @LogVec theta r)
      o1 = argmax1 (op1S @LogVec theta r)
      w = argmax1 (swapS @LogVec theta r) == 1 -- support [False, True]
      o2 = argmax1 (op2S @LogVec theta r)
   in evalSketch perm o1 w o2 ns

-- | Fraction of items whose decoded sketch yields exactly the observed answer.
answerAcc :: Weights -> [WapItem] -> Double
answerAcc theta items =
  let ok = length [() | (p, ns, y) <- items, predictAnswer theta p ns == Just y]
   in fromIntegral ok / fromIntegral (max 1 (length items))

-- | G-layer manifest piece for the Example: honest labeled metrics, one per split.
report :: Weights -> WapDataset -> IO Report
report theta ds =
  return
    ( Report
        [ ("Ans-acc(train)", answerAcc theta (trainItems ds)),
          ("Ans-acc(dev)", answerAcc theta (devItems ds)),
          ("Ans-acc(test)", answerAcc theta (testItems ds))
        ]
    )
