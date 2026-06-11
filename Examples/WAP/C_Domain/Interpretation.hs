{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}

-- | Interpretation I_gamma for WAP. The raw input sort is TEXT (token ids), not a tensor --
--   the trunk symbol 'repF' owns the COLLATION: each problem runs through the reference
--   encoder ('C_Domain.NeuralNets.WapRNN') and the per-problem representations are stacked
--   to @[B, 4096]@. The trunk is a DETERMINISTIC computational Fun symbol; its \eta-lift
--   'repS' is the class DEFAULT (defined once in the signature), so the instances assign
--   only the heads -- each its raw logits as a @LogVec@ leaf over the bound representation:
--
--     repF             = stack . map (gather . bigru . embed)   -- the trunk (Fun, monad-free)
--     permuteS \@LogVec = LogLeaf [0..5]       . head_0          -- [B, 6] raw logits
--     op1S     \@LogVec = LogLeaf [0..3]       . head_1          -- [B, 4]
--     swapS    \@LogVec = LogLeaf [False,True] . head_2          -- [B, 2]
--     op2S     \@LogVec = LogLeaf [0..3]       . head_3          -- [B, 4]
--
--   and the @Dist@ readings are 'decode' of the corresponding leaves on a singleton batch,
--   exactly as @digit \@Dist@ is for MNIST.
module WAP.C_Domain.Interpretation
  ( module WAP.C_Domain.Signature,
    Params,
    initParams,
  )
where

import A_Categorical.Monads.Bridge (decode)
import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogVec (LogVec (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)
import C_Domain.NeuralNets.WapRNN (wapArch, wapHeadLogits, wapRep)
import WAP.C_Domain.Signature
import qualified Torch

-- | The parameter space: the pure weights of the WHOLE model (trunk + four heads), sampled
--   from the one 'wapArch' so the optimizer sees a single \theta.
type Params = Weights

-- | Draw the initial \theta_0.
initParams :: IO Params
initParams = sampleWeights wapArch

-- | The Fun interpretation: 'evalSketch' is the sketch composition on integers, with
--   guarded division (positive divisor, exact quotient) so a failing sketch contributes
--   nothing; 'repF' is the trunk -- collation (encode each problem: embedding -> BiGRU ->
--   8-state gather) and stack to @[B, 4096]@.
instance WapFun where
  evalSketch p o1 w o2 (a, b, c) = do
    let (n1, n2, n3) = permute3 p
    r1 <- applyOp o1 n1 n2
    let (x, y) = if w then (n3, r1) else (r1, n3)
    applyOp o2 x y
    where
      permute3 i = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)] !! i
      applyOp 0 x y = Just (x + y)
      applyOp 1 x y = Just (x - y)
      applyOp 2 x y = Just (x * y)
      applyOp _ x y = if y > 0 && x `mod` y == 0 then Just (x `div` y) else Nothing
  repF theta ps = Torch.stack (Torch.Dim 0) (map (wapRep theta) ps)

-- 'repS' = the class default (the \eta-lift of 'repF'); the instances supply the heads.
instance WapKlFun LogVec where
  permuteS theta r = LogLeaf [0 .. 5] (wapHeadLogits 0 theta r)
  op1S theta r = LogLeaf [0 .. 3] (wapHeadLogits 1 theta r)
  swapS theta r = LogLeaf [False, True] (wapHeadLogits 2 theta r)
  op2S theta r = LogLeaf [0 .. 3] (wapHeadLogits 3 theta r)

instance WapKlFun Dist where
  permuteS theta = decode . permuteS @LogVec theta
  op1S theta = decode . op1S @LogVec theta
  swapS theta = decode . swapS @LogVec theta
  op2S theta = decode . op2S @LogVec theta
