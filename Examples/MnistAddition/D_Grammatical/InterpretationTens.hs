{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

-- | GeomU interpretation of the MNIST-addition axiom, in the TensorProb ([0,1] product
--   fuzzy) logic -- the differentiable mirror of the MeasU reading in
--   "MnistAddition.D_Grammatical.InterpretationData". The SAME 'mnistFormula', read at
--   @\@GeomU@: @digit@ = logits, @plus@ = log-space convolution (the marginalization,
--   kept in logit space), @eqNat@ = softmax + select -> @P(sum=n)@ in [0,1]. The
--   @forall@ over the data batch is the logic's 'bigWedge' (= product t-norm, a
--   geometric mean), so the GeomU reading of the WHOLE axiom is the satisfaction @SAT@
--   that the inference layer (F) then penalizes via @lossKnow@.
module MnistAddition.D_Grammatical.InterpretationTens
  ( mnistSat,
    mnistAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.TensorProb (OmegaP (..))
import B_Logical.Library.SatAgg (satAgg)
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.D_Grammatical.Signature (mnistFormula)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Data.Functor.Identity (Identity (..), runIdentity)
import qualified Torch

-- | The whole axiom in GeomU: @forall@ over the batch = 'bigWedge' (product t-norm,
--   computed as a geometric mean) of the per-pair satisfaction @P(sum=n)@ (which is
--   @mnistFormula \@GeomU@). Mirrors 'MnistAddition.D_Grammatical.InterpretationData.mnistAxiomData'
--   (MeasU). The @1.0@ is the inert @ParamsLogic@ (the product @forall@ is parameter-free).
mnistAxiomTens :: Weights -> (Torch.Tensor, Torch.Tensor, Torch.Tensor) -> OmegaP
mnistAxiomTens theta triple =
  let OmegaP perPair = runIdentity (mnistFormula @GeomU theta triple) -- [B] = P(sum=n)
   in runIdentity (bigWedge @Torch.Tensor @GeomU @OmegaP (1.0 :: Float) perPair (Identity . OmegaP))

-- | The knowledge-base satisfaction exported to the inference layer: the conjunction
--   ('satAgg') of MNIST's closed axioms. MNIST has a single axiom, so this collapses to
--   it -- but it is written as a conjunction so a second axiom is a one-line addition.
mnistSat :: Weights -> (Torch.Tensor, Torch.Tensor, Torch.Tensor) -> OmegaP
mnistSat theta batch = satAgg [mnistAxiomTens theta batch]
