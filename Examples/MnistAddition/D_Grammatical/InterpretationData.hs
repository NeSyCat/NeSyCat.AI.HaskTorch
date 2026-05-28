{-# LANGUAGE TypeApplications #-}

-- | MeasU interpretation of the MNIST-addition axiom (DATA + Dist): the ONE
--   'mnistSentence' read at @\@MeasU@ -- the @Dist@ bind /is/ the law of total
--   probability, so @forall@ over the dataset is the product over @Bool@. Mirrors
--   "MnistAddition.D_Grammatical.InterpretationTens" (the GeomU reading); one formula,
--   two interpretations. This is the probability reading (not differentiable).
module MnistAddition.D_Grammatical.InterpretationData
  ( mnistAxiomData,
  )
where

import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.Interpretations.Boolean () -- A2MonBLat _ MeasU Bool (bigWedge)
import MnistAddition.C_Domain.Interpretation ()
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import qualified Torch

-- | MNIST axiom in MeasU: 'mnistSentence' at @\@MeasU@ over the dataset of pairs.
mnistAxiomData :: Weights -> [(Torch.Tensor, Torch.Tensor, Int)] -> Dist Bool
mnistAxiomData theta dataset = mnistSentence @MeasU () dataset theta
