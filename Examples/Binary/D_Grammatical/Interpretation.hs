{-# LANGUAGE TypeApplications #-}

-- | Grammatical layer (D) — INTERPRETATION for Binary classification: the ONE 'binarySentence'
--   (D/Signature) read in BOTH monads, side by side. One formula, two interpretations -- they
--   differ ONLY in the @Guard@ (the iteration-vs-vectorization axis):
--
--     Dist   (DATA):  the guard is a LIST of points @[Point]@        -> fold over it
--     LogVec (TENS):  the guard is one BATCHED @[B,2]@ tensor        -> one vectorized op
--
--   (Unlike MNIST, there is no observed value passed in: the label is the domain's own circle
--   test, computed inside @labelA@, so the only difference here is the guard shape.) The @Dist@
--   reading is the (non-differentiable) probability reading; the @LogVec@ reading is the 'sat'.
module Binary.D_Grammatical.Interpretation
  ( binaryAxiomData,
    binaryAxiomTens,
  )
where

import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.Category.Monads.LogVec (LogVec)
import B_Logical.Interpretations.Boolean () -- A2MonBLat Dist Bool   (the fold quantifier)
import B_Logical.Interpretations.TensorBool () -- A2MonBLat LogVec Bool (the batched quantifier)
import Binary.C_Domain.Interpretation () -- the BinaryRel / BinaryKlRel instances
import Binary.C_Domain.Signature (Omega, Point)
import Binary.D_Grammatical.Signature (binarySentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- | The @Dist@ reading (probability; not differentiable): the guard is a finite LIST of points,
--   folded one at a time. (Not called at runtime -- the parallel to 'binaryAxiomTens'.)
binaryAxiomData :: Weights -> [Point] -> Dist Omega
binaryAxiomData theta guard = binarySentence @Dist () guard theta

-- | The @LogVec@ reading (differentiable training, the 'sat'): the guard is ONE batched @[B,2]@
--   tensor of points, handled in a single vectorized op.
binaryAxiomTens :: Weights -> Torch.Tensor -> LogVec Omega
binaryAxiomTens theta guard = binarySentence @LogVec () guard theta
