{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in GeomU (TENS + Identity).
module Binary.D_Grammatical.InterpretationTens
  ( binaryAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import Binary.C_Domain.Signature (BinarySorts (..))
import Binary.C_Domain.Interpretation ()
import C_Domain.Models.Interpretations.MLP (MLPSpace)
import B_Logical.LogicalSignature (LogicalSignature (..))
import Binary.D_Grammatical.Signature (binarySentence)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | Binary axiom in GeomU (TENS + Identity).
--   Guard is Torch.Tensor -- a batch tensor (finite subset of the tensor space).
binaryAxiomTens :: Torch.Tensor -> Torch.Tensor -> MLPSpace -> Omega GeomU
binaryAxiomTens betaT guard paramMLP =
  runIdentity (binarySentence @GeomU betaT guard paramMLP)
