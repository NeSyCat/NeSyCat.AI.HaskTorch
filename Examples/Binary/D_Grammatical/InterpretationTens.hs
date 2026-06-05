{-# LANGUAGE TypeApplications #-}

-- | @LogVec@ interpretation of the Binary axiom (TENS + LogVec): the ONE 'binarySentence' read
--   at @\@LogVec@ (the differentiable reading, the satisfaction the inference layer penalizes).
--   Mirrors "Binary.D_Grammatical.InterpretationData" (the @Dist@ reading); one formula, two
--   interpretations. Like MNIST, the satisfaction object is the sentence ITSELF, a
--   @LogVec Bool@: the F layer reads it out to a degree with 'logVecPTrue' and negative-logs
--   it (binary cross-entropy). The crisp Boolean logic is parameter-free, so the logic
--   parameter is @()@.
module Binary.D_Grammatical.InterpretationTens
  ( binaryAxiomTens,
  )
where

import A_Categorical.Category.Monads.LogVec (LogVec)
import B_Logical.Interpretations.TensorBool () -- TwoMonBLat Bool + A2MonBLat LogVec Bool (the quantifier)
import Binary.C_Domain.Interpretation ()
import Binary.C_Domain.Signature (Omega)
import Binary.D_Grammatical.Signature (binarySentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- | Binary axiom in @LogVec@: 'binarySentence' at @\@LogVec@ over the batch tensor (the guard).
binaryAxiomTens :: Weights -> Torch.Tensor -> LogVec Omega
binaryAxiomTens theta guard =
  binarySentence @LogVec () guard theta
