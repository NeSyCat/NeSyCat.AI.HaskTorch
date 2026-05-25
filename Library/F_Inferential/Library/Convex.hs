-- | Convex combination of the two losses:
--   convex(J_data, J_know, lambda) = (1-lambda)*J_data + lambda*J_know.
--   lambda=0: pure data; lambda=1: pure axiom.
module F_Inferential.Library.Convex (convex) where

import qualified Torch

convex :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor -> Torch.Tensor
convex dataLoss knowLoss lambda =
  let oneMinusLambda = Torch.onesLike lambda `Torch.sub` lambda
   in (oneMinusLambda `Torch.mul` dataLoss) `Torch.add` (lambda `Torch.mul` knowLoss)
