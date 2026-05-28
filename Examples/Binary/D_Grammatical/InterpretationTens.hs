{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

-- | GeomU interpretation of the Binary axiom (TENS + Identity): the ONE 'binarySentence'
--   read at @\@GeomU@ (the differentiable reading, the satisfaction the inference layer
--   penalizes). Mirrors "Binary.D_Grammatical.InterpretationData" (the MeasU reading);
--   one formula, two interpretations. @beta = 1.75@ is the logit logic's smoothing.
module Binary.D_Grammatical.InterpretationTens
  ( binaryAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.Tensor () -- TwoMonBLat / A2MonBLat GeomU Torch.Tensor (the logit logic)
import Binary.C_Domain.Signature (BinarySorts (..))
import Binary.C_Domain.Interpretation ()
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Binary.D_Grammatical.Signature (binarySentence)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | Binary axiom in GeomU: 'binarySentence' at @\@GeomU@ over the batch tensor (the guard).
binaryAxiomTens :: Weights -> Torch.Tensor -> Omega GeomU
binaryAxiomTens theta guard =
  runIdentity (binarySentence @GeomU (Torch.asTensor (1.75 :: Float)) guard theta)
