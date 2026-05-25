-- | Binary cross-entropy data loss:  ce(p, y) = -[y*log p + (1-y)*log(1-p)].
module F_Inferential.Library.CrossEntropy (crossEntropy) where

import qualified Torch

crossEntropy :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
crossEntropy p y =
  let logP = Torch.log p
      log1P = Torch.log (Torch.onesLike p `Torch.sub` p)
      yLogP = y `Torch.mul` logP
      y1Log1P = (Torch.onesLike y `Torch.sub` y) `Torch.mul` log1P
   in negate (yLogP `Torch.add` y1Log1P)
