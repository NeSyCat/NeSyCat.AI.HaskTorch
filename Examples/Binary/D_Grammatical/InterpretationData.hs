{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in the @Dist@ monad (DATA + Dist).
module Binary.D_Grammatical.InterpretationData
  ( binaryAxiomData,
  )
where

import A_Categorical.Category.Monads.Dist (Dist)
import B_Logical.Interpretations.Boolean ()
import Binary.C_Domain.Signature (Omega, Point)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Binary.C_Domain.Interpretation ()
import Binary.D_Grammatical.Signature (binarySentence)

-- | Binary axiom in @Dist@ (DATA): 'binarySentence' at @\@Dist@, evaluated probabilistically.
--   Guard is @[Point] = [Torch.Tensor]@ -- a finite subset of R^2 (each a [2] tensor). Mirrors
--   "Binary.D_Grammatical.InterpretationTens" (theta, data).
binaryAxiomData :: Weights -> [Point] -> Dist Omega
binaryAxiomData theta guard = binarySentence @Dist () guard theta
