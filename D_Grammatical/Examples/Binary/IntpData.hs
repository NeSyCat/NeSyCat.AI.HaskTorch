{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in MeasU (DATA + Dist).
module D_Grammatical.Examples.Binary.IntpData
  ( binaryAxiomData,
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.Interpretations.Boolean ()
import C_Domain.Examples.Binary.Signature (BinarySorts (..))
import C_Domain.Models.MLP (ParamsMLP)
import C_Domain.Examples.Binary.Interpretation ()
import D_Grammatical.Examples.Binary.Formulas (binarySentence)

-- | Binary axiom in MeasU (DATA + Dist).
--   Evaluates the formula probabilistically (Mon = Dist).
--   Guard is [Point MeasU] = [(Float, Float)] -- a finite subset of R^2.
binaryAxiomData :: [Point MeasU] -> ParamsMLP -> M MeasU (Omega MeasU)
binaryAxiomData guard paramMLP = binarySentence @MeasU () guard paramMLP
