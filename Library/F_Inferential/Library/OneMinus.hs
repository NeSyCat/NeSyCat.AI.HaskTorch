-- | One-minus penalty:  pen(oneTens, sat) = 1 - sat.
module F_Inferential.Library.OneMinus (oneMinus) where

import qualified Torch

oneMinus :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
oneMinus oneTens sat = oneTens `Torch.sub` sat
