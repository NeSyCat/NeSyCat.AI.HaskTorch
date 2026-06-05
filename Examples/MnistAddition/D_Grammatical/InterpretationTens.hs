{-# LANGUAGE TypeApplications #-}

-- | @LogVec@ interpretation of the MNIST-addition axiom (TENS + LogVec): the ONE
--   'mnistSentence' read at @\@LogVec@ (the differentiable reading). Mirrors
--   "MnistAddition.D_Grammatical.InterpretationData" (the @Dist@ reading) — one formula,
--   two interpretations. The satisfaction object is the sentence ITSELF, a @LogVec Bool@
--   (the sibling of @Dist Bool@): the F layer reads it out to a degree with
--   'logVecPTrue' (the twin of @distPTrue@), then penalizes it. The crisp Boolean logic is
--   parameter-free, so the logic parameter is @()@. The @LogVec@ meaning of the symbols comes
--   from the C interpretation (digit/plus/eqNat) and the B interpretation (the product
--   'bigWedge' for @Bool@).
module MnistAddition.D_Grammatical.InterpretationTens
  ( mnistAxiomTens,
  )
where

import A_Categorical.Category.Bridge (encodeBatch)
import A_Categorical.Category.Monads.LogVec (LogVec)
import B_Logical.Interpretations.TensorBool () -- the A2MonBLat LogVec Bool quantifier instance
import MnistAddition.C_Domain.Interpretation () -- the MnistKlRel LogVec instance
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- | MNIST axiom in @LogVec@: 'mnistSentence' at @\@LogVec@ — the satisfaction object the
--   inference layer reads out and penalizes. @batch@ is the guard (the batched
--   @(image,image,sum)@ triple); the observed one-hot sum is lifted into the monad by the
--   batched bridge 'encodeBatch' over support @[0..18]@ (the @encode@ of the observation).
mnistAxiomTens :: Weights -> (Torch.Tensor, Torch.Tensor, Torch.Tensor) -> LogVec Bool
mnistAxiomTens theta (x, y, oneHotN) =
  mnistSentence @LogVec () (x, y, encodeBatch [0 .. 18] oneHotN) theta
