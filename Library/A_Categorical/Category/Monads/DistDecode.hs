-- | Decoding a neural output into the MeasU distribution monad: the modular softmax. The net
--   emits PURE LOGITS; this is the only place a softmax appears on the MeasU side, kept as one
--   reusable function so the relation symbols are pure compositions, e.g.
--
--     @digit \@MeasU theta = categorical [0..9] . cnn theta@
--     @classifierA \@MeasU theta = categorical [True,False] . mlp theta@
--
--   The GeomU sibling needs no function: it is just @LogLeaf support@ (the raw logits as a
--   'A_Categorical.Category.Monads.LogVec.LogVec' leaf), and the softmax there is implicit in
--   the log-space readout, never in training.
module A_Categorical.Category.Monads.DistDecode (categorical) where

import A_Categorical.Category.Monads.Dist (Dist (..))
import qualified Torch
import qualified Torch.Functional as F

-- | @categorical support logits@: decode one row of logits (@[1,k]@, @k = length support@)
--   into a categorical 'Dist' over @support@ by softmax. The modular MeasU reading of a neural
--   output -- the softmax that used to be inlined in @digit@/@classifierA@.
categorical :: [a] -> Torch.Tensor -> Dist a
categorical support logits =
  let ps = head (Torch.asValue (F.softmax (F.Dim 1) logits) :: [[Float]])
   in FiniteSupp [(x, realToFrac (ps !! i)) | (i, x) <- zip [0 ..] support]
