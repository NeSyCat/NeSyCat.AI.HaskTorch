{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in MeasU (DATA + Dist).
module Binary.D_Grammatical.InterpretationData
  ( binaryAxiomData,
  )
where

import A_Categorical.CategoricalSignature (Framework (..))
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.Interpretations.Boolean ()
import Binary.C_Domain.Signature (BinarySorts (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Binary.C_Domain.Interpretation ()
import Binary.D_Grammatical.Signature (binarySentence)

-- | Binary axiom in MeasU (DATA + Dist): 'binarySentence' at @\@MeasU@, evaluated
--   probabilistically (M = Dist). Guard is @[Point MeasU] = [(Float, Float)]@ -- a finite
--   subset of R^2. Mirrors "Binary.D_Grammatical.InterpretationTens" (theta, data).
binaryAxiomData :: Weights -> [Point MeasU] -> M MeasU (Omega MeasU)
binaryAxiomData theta guard = binarySentence @MeasU () guard theta
