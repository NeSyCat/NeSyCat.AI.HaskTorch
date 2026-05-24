{-# LANGUAGE TypeApplications #-}

-- | GeomU interpretation of the MNIST-addition formula: the 'Identity' monad in
--   tensor spaces. The same 'mnistFormula', read at @\@GeomU@ -- @digit@ is raw
--   logits and @plus@ is the log-space convolution (LogSumExp over @d1+d2=s@), so
--   each pair's satisfaction is a logit; the @forall@ is the logic's smooth-inf
--   'bigWedge'. Differentiable on logits (no softmax) -- this is the training signal.
module D_Grammatical.Examples.MnistAddition.IntpTens
  ( mnistAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.Tensor () -- LogicalQuantSignature Tensor GeomU Omega (bigWedge)
import B_Logical.LogicalQuantSignature (LogicalQuantSignature (..))
import C_Domain.Examples.MnistAddition.Interpretation ()
import C_Domain.Models.MnistCNN (ParamsCNN)
import D_Grammatical.Examples.MnistAddition.Formulas (mnistFormula)
import Data.Functor.Identity (Identity (..), runIdentity)
import qualified Torch

-- | The MNIST axiom in GeomU. @betaT@ is the logic smoothing parameter; the batch
--   is (images x @[B,1,28,28]@, images y @[B,1,28,28]@, one-hot observed sums
--   @[B,19]@). 'mnistFormula' runs once on the batch giving per-pair satisfaction
--   logits @[B]@, then the universal quantifier (smooth-inf) reduces them to @[1]@.
mnistAxiomTens ::
  Torch.Tensor ->
  (Torch.Tensor, Torch.Tensor, Torch.Tensor) ->
  ParamsCNN ->
  Torch.Tensor
mnistAxiomTens betaT batch theta =
  let sat = runIdentity (mnistFormula @GeomU theta batch) -- [B] per-pair logit truths
   in runIdentity (bigWedge betaT sat Identity) -- forall = smooth-inf -> [1]
