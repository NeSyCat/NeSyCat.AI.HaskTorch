{-# LANGUAGE TypeApplications #-}

-- | GeomU interpretation of the MNIST-addition formula (Identity monad, tensor
--   spaces). The same symbols, read at @\@GeomU@: @digit@ is raw logits and
--   @plus@ is the log-space convolution, so the /term/ @digit(x) + digit(y)@
--   interprets to a logit vector over sums @[B,19]@ -- pure logits, no softmax.
--
--   The remaining symbols are interpreted per universe in different layers: in
--   MeasU @eqNat@/@(=)@ gives @P(sum=n)@ (via the bridge's softmax); in GeomU the
--   @= add(x,y)@ and the @forall@ over pairs are realized by MNIST's /inference/
--   interpretation (categorical NLL), keeping the softmax out of GeomU entirely.
module Examples.MnistAddition.D_Grammatical.InterpretationTens
  ( mnistSumLogits,
  )
where

import Lib.A_Categorical.CategoricalInterpretation (GeomU)
import Examples.MnistAddition.C_Domain.Interpretation ()
import Examples.MnistAddition.C_Domain.Signature (MnistArith (..), MnistKlRel (..))
import Lib.C_Domain.Models.MnistCNN (ParamsCNN)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | @digit(x) + digit(y)@ in GeomU: a logit vector over the 19 possible sums,
--   @[B,1,28,28] x [B,1,28,28] -> [B,19]@, built from @digit \@GeomU@ (logits)
--   and @plus \@GeomU@ (log-space convolution). No softmax, no normalization.
mnistSumLogits :: ParamsCNN -> (Torch.Tensor, Torch.Tensor) -> Torch.Tensor
mnistSumLogits theta (xB, yB) =
  plus @GeomU
    (runIdentity (digit @GeomU theta xB))
    (runIdentity (digit @GeomU theta yB))
