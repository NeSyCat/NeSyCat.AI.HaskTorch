{-# LANGUAGE TypeApplications #-}

-- | GeomU interpretation of the MNIST-addition axiom (TENS + Identity): the ONE
--   'mnistSentence' read at @\@GeomU@ (the differentiable reading). Mirrors
--   "MnistAddition.D_Grammatical.InterpretationData" (the MeasU reading) — one formula,
--   two interpretations. The product-t-norm logic is parameter-free, so the logic
--   parameter is @()@. The GeomU meaning of the symbols comes from the C interpretation
--   (digit/plus/eqNat) and the B interpretation (the product 'bigWedge' for @OmegaP@).
module MnistAddition.D_Grammatical.InterpretationTens
  ( mnistAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.TensorProb (OmegaP)
import MnistAddition.C_Domain.Interpretation ()
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | MNIST axiom in GeomU: 'mnistSentence' at @\@GeomU@, the satisfaction the inference
--   layer penalizes. @batch@ is the guard (the batched @(image,image,sum)@ triple).
mnistAxiomTens :: Weights -> (Torch.Tensor, Torch.Tensor, Torch.Tensor) -> OmegaP
mnistAxiomTens theta batch = runIdentity (mnistSentence @GeomU () batch theta)
