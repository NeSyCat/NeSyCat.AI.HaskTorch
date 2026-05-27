{-# LANGUAGE TypeApplications #-}

-- | GeomU interpretation of the MNIST-addition formula (Identity monad, tensor
--   spaces). The same symbols, read at @\@GeomU@: @digit@ is raw logits and
--   @plus@ is the log-space convolution, so the /term/ @digit(x) + digit(y)@
--   interprets to a logit vector over sums @[B,19]@ -- pure logits, no softmax.
--
--   The remaining symbols are interpreted per universe in different layers: in
--   MeasU @eqNat@/@(=)@ gives @P(sum=n)@ (via the bridge's softmax); in GeomU the
--   @= add(x,y)@ (softmax + select n -> @P(sum=n)@) and the @forall@ over pairs
--   (the logic's 'bigWedge' read as p-mean-error) are realized by MNIST's /inference/
--   interpretation (@J = 1 - SAT@), keeping the softmax out of GeomU entirely.
module MnistAddition.D_Grammatical.InterpretationTens
  ( mnistSumLogits,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.C_Domain.Signature (MnistArith (..), MnistKlRel (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | @digit(x) + digit(y)@ in GeomU: a logit vector over the 19 possible sums,
--   @[B,1,28,28] x [B,1,28,28] -> [B,19]@, built from @digit \@GeomU@ (logits)
--   and @plus \@GeomU@ (log-space convolution). No softmax, no normalization.
mnistSumLogits :: Weights -> (Torch.Tensor, Torch.Tensor) -> Torch.Tensor
mnistSumLogits theta (xB, yB) =
  plus @GeomU
    (runIdentity (digit @GeomU theta xB))
    (runIdentity (digit @GeomU theta yB))
