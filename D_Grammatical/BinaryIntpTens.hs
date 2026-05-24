{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in GeomU (TENS + Identity).
module D_Grammatical.BinaryIntpTens
  ( binaryAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import C_Domain.BinarySignature (BinarySorts (..))
import C_Domain.BinaryInterpretation ()
import C_Domain.Models.MLP (ParamsMLP)
import B_Logical.LogicalSignature (LogicalSignature (..))
import D_Grammatical.BinaryFormulas (binarySentence)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | Binary axiom in GeomU (TENS + Identity).
--   Guard is Torch.Tensor -- a batch tensor (finite subset of the tensor space).
binaryAxiomTens :: Torch.Tensor -> Torch.Tensor -> ParamsMLP -> Omega GeomU
binaryAxiomTens betaT guard paramMLP =
  runIdentity (binarySentence @GeomU betaT guard paramMLP)
