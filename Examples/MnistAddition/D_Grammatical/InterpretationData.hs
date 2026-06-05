{-# LANGUAGE TypeApplications #-}

-- | @Dist@ interpretation of the MNIST-addition axiom (DATA + Dist): the ONE
--   'mnistSentence' read at @\@Dist@ -- the @Dist@ bind /is/ the law of total
--   probability, so @forall@ over the dataset is the product over @Bool@. Mirrors
--   "MnistAddition.D_Grammatical.InterpretationTens" (the @LogVec@ reading); one formula,
--   two interpretations. This is the probability reading (not differentiable).
module MnistAddition.D_Grammatical.InterpretationData
  ( mnistAxiomData,
  )
where

import A_Categorical.Category.Monads.Dist (Dist)
import B_Logical.Interpretations.Boolean () -- A2MonBLat _ Dist Bool (bigWedge)
import MnistAddition.C_Domain.Interpretation ()
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import qualified Torch

-- | MNIST axiom in @Dist@: 'mnistSentence' at @\@Dist@ over the dataset of pairs. The
--   observed sum @n@ enters the formula as a (certain) monadic value @pure n :: Dist Int@,
--   so the formula is identical to the @LogVec@ reading -- one formula, two interpretations.
mnistAxiomData :: Weights -> [(Torch.Tensor, Torch.Tensor, Int)] -> Dist Bool
mnistAxiomData theta dataset =
  mnistSentence @Dist () [(x, y, pure n) | (x, y, n) <- dataset] theta
