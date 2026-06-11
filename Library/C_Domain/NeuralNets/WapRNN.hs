-- | The WAP sentence encoder (the reference architecture of differentiable-Forth /
--   DeepProbLog / DeepStochLog):
--
--     Embedding(746, 256) -> 1-layer BiGRU(512) ->
--     concat [ fwd state at the LAST token, fwd states at the 3 number positions,
--              bwd state at the FIRST token, bwd states at the 3 number positions ]  (8 x 512)
--     -> four linear heads (6 / 4 / 2 / 4): permute, op1, swap, op2.
--
--   Unlike the MNIST CNN this net is NOT a pipeline, so it is more than one 'Arch' +
--   'runArch': (i) it is a DAG -- one trunk fanning out into four heads, giving the symbol
--   set five entry points (the trunk 'wapRep' and the heads 'wapHeadLogits'); (ii) the
--   8-state gather is INPUT-dependent (the number positions are data, not architecture),
--   so it lives between the segments as glue no fixed @Layer@ can express. The model is
--   still ONE architecture ('wapArch' = the concatenated segments, one \theta for the
--   optimizer); 'segmentWeights' deals \theta out to the segments.
module C_Domain.NeuralNets.WapRNN
  ( wapArch,
    wapHeadSizes,
    wapRep,
    wapHeadLogits,
  )
where

import C_Domain.NeuralNets.DSL.Semantics (Weights, runArch, segmentWeights)
import C_Domain.NeuralNets.DSL.Syntax (Arch, Layer (..))
import qualified Torch

-- | The four head widths: permute (6 orderings), op1 (4 ops), swap (2), op2 (4 ops).
wapHeadSizes :: [Int]
wapHeadSizes = [6, 4, 2, 4]

-- | The model's sequential SEGMENTS: the trunk (embedding, BiGRU), then the four heads.
wapSegs :: [Arch]
wapSegs = [Embedding 746 256] : [BiGRU 256 512] : [[Linear (8 * 512) k] | k <- wapHeadSizes]

-- | The whole model as ONE architecture (one \theta for the optimizer).
wapArch :: Arch
wapArch = concat wapSegs

-- | The trunk forward at \theta: ONE problem (token ids + the positions of the three
--   @\<NR\>@ tokens) |-> its @[4096]@ representation (the 8-state concatenation).
wapRep :: Weights -> ([Int], [Int]) -> Torch.Tensor
wapRep theta (toks, nrPos) =
  Torch.cat (Torch.Dim 0) ([state 0 i | i <- (l - 1) : nrPos] ++ [state 1 i | i <- 0 : nrPos])
  where
    [embW, gruW] = take 2 (segmentWeights wapSegs theta)
    l = length toks
    e = runArch (head wapSegs) embW (Torch.asTensor toks) -- [L, 256]
    out = runArch (wapSegs !! 1) gruW (Torch.reshape [1, l, 256] e) -- [1, L, 1024]
    o = Torch.reshape [l, 2, 512] out -- [L, dir, 512] (0 = fwd, 1 = bwd)
    state d i = Torch.reshape [512] (Torch.sliceDim 1 d (d + 1) 1 (Torch.sliceDim 0 i (i + 1) 1 o))

-- | Head @i@'s forward at \theta: @[B, 4096] -> [B, k_i]@ raw logits.
wapHeadLogits :: Int -> Weights -> Torch.Tensor -> Torch.Tensor
wapHeadLogits i theta = runArch (wapSegs !! (2 + i)) (segmentWeights wapSegs theta !! (2 + i))
