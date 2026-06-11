-- | Parameterized: a 1-layer bidirectional GRU. Parameter space, per direction:
--   w_{ih} : \mathbb{R}^{3h \times i}, w_{hh} : \mathbb{R}^{3h \times h},
--   b_{ih}, b_{hh} : \mathbb{R}^{3h} -- 8 tensors total, kept in the flat ATen order
--   @[w_ih, w_hh, b_ih, b_hh]@ forward direction first, then reverse.
--   'sampleBiGRU' draws fresh \theta uniform in (-k, k) with k = 1/\sqrt{h}
--   (torch.nn.GRU's init); 'bigru' is the forward @[B, L, i] -> [B, L, 2h]@ via the fused
--   ATen @gru@ kernel -- the SAME op @torch.nn.GRU@ dispatches to, so a reference PyTorch
--   architecture (e.g. DeepProbLog's WAP encoder) ports verbatim, weight layout and all.
module C_Domain.NeuralNets.DSL.Library.Parameterized.BiGRU (BiGRUW (..), sampleBiGRU, bigru) where

import Torch (Parameter, Parameterized (..), Tensor, makeIndependent, toDependent)
import qualified Torch
import qualified Torch.Functional.Internal as FI

-- | The 8 GRU parameter tensors, in flat ATen order (forward dir, then reverse).
newtype BiGRUW = BiGRUW [Parameter] deriving (Show)

instance Parameterized BiGRUW where
  flattenParameters (BiGRUW ps) = ps
  _replaceParameters (BiGRUW ps) = BiGRUW <$> mapM _replaceParameters ps

sampleBiGRU :: Int -> Int -> IO BiGRUW
sampleBiGRU i h = do
  let k = 1.0 / sqrt (fromIntegral h) :: Float
      draw shp = do
        r <- Torch.randIO' shp -- uniform [0,1)
        return ((r `Torch.mul` Torch.asTensor (2 * k)) `Torch.sub` Torch.asTensor k)
  ts <-
    mapM
      draw
      [ [3 * h, i], [3 * h, h], [3 * h], [3 * h], -- forward:  w_ih, w_hh, b_ih, b_hh
        [3 * h, i], [3 * h, h], [3 * h], [3 * h] -- reverse:  w_ih, w_hh, b_ih, b_hh
      ]
  BiGRUW <$> mapM makeIndependent ts

bigru :: BiGRUW -> Tensor -> Tensor
bigru (BiGRUW ps) x =
  let hsz = Torch.shape (toDependent (ps !! 1)) !! 1 -- w_hh : [3h, h]
      b = head (Torch.shape x)
      h0 = Torch.zeros' [2, b, hsz] -- 1 layer x 2 directions
      (out, _) = FI.gru x h0 (map toDependent ps) True 1 0.0 False True True
   in out -- [B, L, 2h]: out[.., 0:h] forward states, out[.., h:2h] reverse states
