{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in GeomU (TENS + Identity).
module Examples.Binary.D_Grammatical.IntpTens
  ( binaryAxiomTens,
  )
where

import Lib.A_Categorical.CategoricalInterpretation (GeomU)
import Examples.Binary.C_Domain.Signature (BinarySorts (..))
import Examples.Binary.C_Domain.Interpretation ()
import Lib.C_Domain.Models.MLP (ParamsMLP)
import Lib.B_Logical.LogicalSignature (LogicalSignature (..))
import Examples.Binary.D_Grammatical.Formulas (binarySentence)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | Binary axiom in GeomU (TENS + Identity).
--   Guard is Torch.Tensor -- a batch tensor (finite subset of the tensor space).
binaryAxiomTens :: Torch.Tensor -> Torch.Tensor -> ParamsMLP -> Omega GeomU
binaryAxiomTens betaT guard paramMLP =
  runIdentity (binarySentence @GeomU betaT guard paramMLP)
