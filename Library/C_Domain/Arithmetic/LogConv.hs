-- | Domain arithmetic on classifier logit-distributions, shared across examples.
--
--   'logConv' is the GeomU interpretation of the @plus@ function symbol: the
--   convolution that realizes the law of total probability for the SUM of two digit
--   distributions. It is the geometric shadow of what the MeasU @Dist@-bind does for
--   free (@P(d1+d2=s) = \Sigma_{i+j=s} P(d1=i) P(d2=j)@), so the addition axiom needs
--   no existential -- the @\Sigma@ lives here, inside @plus@. It works on raw logits
--   (LogSumExp of @oplus@-added logits), keeping GeomU inside the logits; softmax happens only
--   later in the inference penalty.
module C_Domain.Arithmetic.LogConv (logConv) where

import qualified Torch
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
