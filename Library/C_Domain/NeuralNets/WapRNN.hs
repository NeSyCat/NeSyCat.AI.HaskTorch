-- | The WAP sentence encoder -- a 1:1 port of DeepProbLog's @wap_network.py@ (Apache-2.0,
--   github.com/ML-KULeuven/deepproblog, examples/Forth/WAP), the reference architecture shared
--   by differentiable-Forth (Bosnjak et al. 2017), DeepProbLog (Manhaeve et al. 2018/2021) and
--   DeepStochLog (Winters et al. 2022):
--
--     Embedding(746, 256) -> 1-layer BiGRU(512) ->
--     concat [ fwd state at the LAST token, fwd states at the 3 number positions,
--              bwd state at the FIRST token, bwd states at the 3 number positions ]  (8 x 512)
--     -> four linear heads (6 / 4 / 2 / 4): permute, op1, swap, op2.
--
--   The whole model is ONE 'Arch' (so one \theta for the optimizer); the non-sequential
--   shape (shared trunk, four heads) is run piecewise via 'splitWeights' + 'runArch'.
module C_Domain.NeuralNets.WapRNN
  ( wapArch,
    wapRep,
    wapHeadLogits,
    wapHeadSizes,
  )
where

import C_Domain.NeuralNets.DSL.Semantics (Weights, runArch, splitWeights)
import C_Domain.NeuralNets.DSL.Syntax (Arch, Layer (..), (>>>))
import qualified Torch

-- | Trunk pieces (the vocab has 746 entries; ids = line numbers, as in the reference tokenizer).
wapEmbedArch, wapGruArch :: Arch
wapEmbedArch = [Embedding 746 256]
wapGruArch = [BiGRU 256 512]

-- | The four head widths: permute (6 orderings), op1 (4 ops), swap (2), op2 (4 ops).
wapHeadSizes :: [Int]
wapHeadSizes = [6, 4, 2, 4]

-- | The whole model as ONE architecture: trunk, then the four heads in order.
wapArch :: Arch
wapArch = wapEmbedArch >>> wapGruArch >>> [Linear (8 * 512) k | k <- wapHeadSizes]

-- | The 8-state sentence representation of ONE problem: token ids + the positions of the
--   three @\<NR\>@ tokens |-> a @[4096]@ vector (the reference's @x1 ++ x2@ concatenation).
wapRep :: Weights -> ([Int], [Int]) -> Torch.Tensor
wapRep theta (toks, nrPos) =
  let (embW, rest) = splitWeights wapEmbedArch theta
      (gruW, _) = splitWeights wapGruArch rest
      l = length toks
      e = runArch wapEmbedArch embW (Torch.asTensor toks) -- [L, 256]
      out = runArch wapGruArch gruW (Torch.reshape [1, l, 256] e) -- [1, L, 1024]
      o = Torch.reshape [l, 2, 512] out -- [L, dir, 512] (0 = fwd, 1 = bwd)
      grab d i = Torch.reshape [512] (Torch.sliceDim 1 d (d + 1) 1 (Torch.sliceDim 0 i (i + 1) 1 o))
      [n1, n2, n3] = nrPos
      x1 = Torch.cat (Torch.Dim 0) [grab 0 (l - 1), grab 0 n1, grab 0 n2, grab 0 n3]
      x2 = Torch.cat (Torch.Dim 0) [grab 1 0, grab 1 n1, grab 1 n2, grab 1 n3]
   in Torch.cat (Torch.Dim 0) [x1, x2] -- [4096]

-- | Head @i@ over a batch of representations: @[B, 4096] -> [B, k_i]@ raw logits.
--   Walks the weights past the trunk and the heads before @i@ ('splitWeights'), then runs
--   head @i@'s one-layer architecture.
wapHeadLogits :: Int -> Weights -> Torch.Tensor -> Torch.Tensor
wapHeadLogits i theta rep =
  let (_, rest1) = splitWeights wapEmbedArch theta
      (_, rest2) = splitWeights wapGruArch rest1
      headArch j = [Linear (8 * 512) (wapHeadSizes !! j)]
      go j ws
        | j == i = fst (splitWeights (headArch i) ws)
        | otherwise = go (j + 1) (snd (splitWeights (headArch j) ws))
   in runArch (headArch i) (go 0 rest2) rep
