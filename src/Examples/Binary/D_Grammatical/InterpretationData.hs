{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in MeasU (DATA + Dist).
module Examples.Binary.D_Grammatical.InterpretationData
  ( binaryAxiomData,
  )
where

import Lib.A_Categorical.CategoricalSignature (Universe (..))
import Lib.A_Categorical.CategoricalInterpretation (MeasU)
import Lib.B_Logical.Interpretations.Boolean ()
import Examples.Binary.C_Domain.Signature (BinarySorts (..))
import Lib.C_Domain.Models.MLP (ParamsMLP)
import Examples.Binary.C_Domain.Interpretation ()
import Examples.Binary.D_Grammatical.Formulas (binarySentence)

-- | Binary axiom in MeasU (DATA + Dist).
--   Evaluates the formula probabilistically (Mon = Dist).
--   Guard is [Point MeasU] = [(Float, Float)] -- a finite subset of R^2.
binaryAxiomData :: [Point MeasU] -> ParamsMLP -> M MeasU (Omega MeasU)
binaryAxiomData guard paramMLP = binarySentence @MeasU () guard paramMLP
