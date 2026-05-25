-- | Negative-log penalty:  pen(p) = -log(p).
module F_Inferential.Library.NegLog (negLog) where

import qualified Torch

negLog :: Torch.Tensor -> Torch.Tensor
negLog p = negate (Torch.log p)
