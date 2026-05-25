{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in MeasU (DATA + Dist).
module Binary.D_Grammatical.InterpretationData
  ( binaryAxiomData,
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.Interpretations.Boolean ()
import Binary.C_Domain.Signature (BinarySorts (..))
import C_Domain.Models.MLP (ParamsMLP)
import Binary.C_Domain.Interpretation ()
import Binary.D_Grammatical.Signature (binarySentence)

-- | Binary axiom in MeasU (DATA + Dist).
--   Evaluates the formula probabilistically (Mon = Dist).
--   Guard is [Point MeasU] = [(Float, Float)] -- a finite subset of R^2.
binaryAxiomData :: [Point MeasU] -> ParamsMLP -> M MeasU (Omega MeasU)
binaryAxiomData guard paramMLP = binarySentence @MeasU () guard paramMLP
