{-# LANGUAGE TypeApplications #-}

-- | GeomU reading of the MNIST knowledge base: the ONE 'mnistSentence' interpreted at
--   @\@GeomU@ (the differentiable reading), conjoined ('satAgg') over the example's axioms
--   (here a single one). The MeasU reading is the SAME 'mnistSentence' at @\@MeasU@ (see
--   "MnistAddition.D_Grammatical.InterpretationData") -- one formula, two interpretations,
--   no per-universe re-definition. The GeomU meaning of the symbols comes entirely from
--   the C interpretation (digit/plus/eqNat) and the B interpretation (the product-t-norm
--   'bigWedge' for @OmegaP@, in "B_Logical.Interpretations.TensorProb").
module MnistAddition.D_Grammatical.InterpretationTens
  ( mnistSat,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.TensorProb (OmegaP)
import B_Logical.Library.SatAgg (satAgg)
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | The knowledge-base satisfaction the inference layer penalizes: the conjunction
--   ('satAgg') of the example's closed axioms, each 'mnistSentence' read at @\@GeomU@.
mnistSat :: Weights -> (Torch.Tensor, Torch.Tensor, Torch.Tensor) -> OmegaP
mnistSat theta batch = satAgg [runIdentity (mnistSentence @GeomU () batch theta)]
