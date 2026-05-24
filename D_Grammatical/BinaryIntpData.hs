{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in MeasU (DATA + Dist).
module D_Grammatical.BinaryIntpData
  ( binaryAxiomData,
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import A_Categorical.CategoricalInterpretation (MeasU)
import B_Logical.Interpretations.Boolean ()
import C_Domain.BinarySignature (BinarySorts (..))
import C_Domain.Models.MLP (ParamsMLP)
import C_Domain.BinaryInterpretation ()
import D_Grammatical.BinaryFormulas (binarySentence)

-- | Binary axiom in MeasU (DATA + Dist).
--   Evaluates the formula probabilistically (Mon = Dist).
--   Guard is [Point MeasU] = [(Float, Float)] -- a finite subset of R^2.
binaryAxiomData :: [Point MeasU] -> ParamsMLP -> M MeasU (Omega MeasU)
binaryAxiomData guard paramMLP = binarySentence @MeasU () guard paramMLP
