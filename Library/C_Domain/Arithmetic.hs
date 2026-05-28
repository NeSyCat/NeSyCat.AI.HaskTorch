-- | GeomU interpretations of domain arithmetic / relation symbols on classifier
--   logit-distributions, shared across examples (the geometric shadows of operations
--   the MeasU @Dist@ monad does for free):
--
--     * 'logConv' — the @plus@ (@+@) symbol: the convolution realizing the law of total
--       probability for the SUM of two digit distributions. Works on raw logits
--       (LogSumExp), keeping GeomU in logit space; no existential needed.
--     * 'probEq'  — the @eqNat@ (@=@) symbol: a soft categorical equality, the
--       probability the (softmax of the) predicted logits assigns to an observed one-hot
--       value, in [0,1]. The bridge from logits to the fuzzy truth (the GeomU analogue of
--       MeasU's @decDigit@). Reusable wherever a categorical @=@ is read as a probability.
module C_Domain.Arithmetic (logConv, probEq) where

import qualified Torch
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI

-- | Log-space convolution of two digit-logit vectors, @[B,10] -> [B,10] -> [B,19]@:
--   @out[b,s] = LogSumExp_{i+j=s} (lx[b,i] + ly[b,j])@. The same LogSumExp/add the
--   TensReal logic uses; none introduced.
logConv :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
logConv lx ly =
  let b = head (Torch.shape lx)
      jj = Torch.reshape [b, 10, 1] lx `Torch.add` Torch.reshape [b, 1, 10] ly -- [B,10,10]
      entry i j = FI.select (FI.select jj 1 i) 1 j -- jj[:,i,j] :: [B]
      colFor s =
        let es = [entry i (s - i) | i <- [0 .. 9], s - i >= 0, s - i <= 9]
         in FI.logsumexp (Torch.stack (Torch.Dim 1) es) 1 False -- [B]
   in Torch.stack (Torch.Dim 1) [colFor s | s <- [0 .. 18]] -- [B,19]

-- | Soft categorical equality, @[B,K] one-hot target -> [B,K] logits -> [B]@:
--   @probEq oneHot logits = <oneHot, softmax(logits)> = P(prediction = observed)@ in [0,1].
--   The only softmax; the marginalization in 'logConv' stays in logit space.
probEq :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor
probEq oneHot logits =
  Torch.sumDim (Torch.Dim 1) Torch.RemoveDim Torch.Float (oneHot `Torch.mul` F.softmax (F.Dim 1) logits)
