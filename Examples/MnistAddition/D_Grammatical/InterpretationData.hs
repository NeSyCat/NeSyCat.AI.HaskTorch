{-# LANGUAGE TypeApplications #-}

-- | MeasU interpretation of the MNIST-addition formula: the 'Dist' monad in
--   measurable spaces. The same 'mnistFormula', read at @\@MeasU@ -- the @Dist@
--   bind /is/ the law of total probability, so @P(sum = n)@ and the product over
--   pairs (the @forall@, via the reused 'bigWedge') both fall out. This is the
--   probability reading (not differentiable); training uses the GeomU side.
module MnistAddition.D_Grammatical.InterpretationData
  ( mnistProb,
    mnistAxiomData,
  )
where

import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.Category.Monads.DistExpect (distPTrue)
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.Interpretations.Boolean () -- LogicalQuantSignature _ MeasU Bool (bigWedge)
import B_Logical.LogicalQuantSignature (LogicalQuantSignature (..))
import MnistAddition.C_Domain.Interpretation ()
import C_Domain.Models.MnistCNN (ParamsCNN)
import MnistAddition.D_Grammatical.Formulas (mnistFormula)
import qualified Torch

-- | @P(add(x,y) = digit(x)+digit(y))@ for one pair: the per-pair satisfaction,
--   read off the @Dist Bool@ via the canonical @Dist(Bool) -> [0,1]@.
mnistProb :: ParamsCNN -> (Torch.Tensor, Torch.Tensor, Int) -> Double
mnistProb theta triple = distPTrue (mnistFormula @MeasU theta triple)

-- | The whole axiom in MeasU: @forall@ over the dataset = 'bigWedge' (= product
--   in @Dist@ over @Bool@), reusing the domain-independent quantifier as-is.
mnistAxiomData :: ParamsCNN -> [(Torch.Tensor, Torch.Tensor, Int)] -> Dist Bool
mnistAxiomData theta dataset = bigWedge () dataset (mnistFormula @MeasU theta)
