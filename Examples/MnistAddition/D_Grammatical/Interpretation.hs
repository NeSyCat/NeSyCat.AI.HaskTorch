{-# LANGUAGE TypeApplications #-}

-- | Grammatical layer (D) — INTERPRETATION for MNIST-addition: the ONE 'mnistSentence'
--   (D/Signature) read in BOTH monads, side by side. One formula, two interpretations -- they
--   differ ONLY along the iteration-vs-vectorization axis (the @Guard@ + the observation):
--
--                      the guard (the data)         the observed sum n
--     Dist   (DATA):   a list  @[(x,y,n)]@ -> fold   @pure n@      (a point mass, per element)
--     LogVec (TENS):   one batch @(xs,ys,ns)@-> mean 'encodeObs'   (a batched delta leaf)
--
--   It is the SAME observed sum n in both -- only its REPRESENTATION differs (the @encode@ of a
--   CERTAIN value: @pure@ when iterating, a batched one-hot leaf when vectorizing). The @Dist@
--   reading is the (non-differentiable) probability reading; the @LogVec@ reading is the
--   satisfaction 'sat' the inference layer trains on.
module MnistAddition.D_Grammatical.Interpretation
  ( mnistAxiomData,
    mnistAxiomTens,
  )
where

import A_Categorical.Category.Bridge (encodeBatch)
import A_Categorical.Category.Monads.Dist (Dist)
import A_Categorical.Category.Monads.LogVec (LogVec)
import B_Logical.Interpretations.Boolean () -- A2MonBLat Dist Bool   (the fold quantifier)
import B_Logical.Interpretations.TensorBool () -- A2MonBLat LogVec Bool (the batched quantifier)
import MnistAddition.C_Domain.Interpretation () -- the MnistKlRel instances
import MnistAddition.C_Domain.Signature (Image, Natural) -- the sorts
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- | The @Dist@ reading (probability; not differentiable): ITERATE the dataset of triples; the
--   observed sum @n@ enters as the point mass @pure n@. (Not called at runtime -- the parallel
--   to 'mnistAxiomTens'; the report reads per-pair via 'mnistFormula' directly.)
mnistAxiomData :: Weights -> [(Image, Image, Natural)] -> Dist Bool
mnistAxiomData theta dataset =
  mnistSentence @Dist () [(x, y, pure n) | (x, y, n) <- dataset] theta

-- | The @LogVec@ reading (differentiable training, the 'sat'): ONE batched triple; the observed
--   sums @ns@ (a @[B]@ tensor of indices) enter as the batched delta 'encodeObs'.
mnistAxiomTens :: Weights -> (Image, Image, Torch.Tensor) -> LogVec Bool
mnistAxiomTens theta (xs, ys, ns) =
  mnistSentence @LogVec () (xs, ys, encodeObs ns) theta

-- | Lift the observed sum(s) into @LogVec@ as a CERTAIN batched distribution -- the @encode@ of
--   the observation, vectorized: the @[B]@ sum indices -> a @[B,19]@ one-hot -> the batched bridge
--   'encodeBatch' (a delta leaf over @[0..18]@). The @LogVec@ analogue of @Dist@'s @pure n@, one
--   row per batch element (this is the unbatched @encode . pure@ done all-at-once for the batch).
encodeObs :: Torch.Tensor -> LogVec Natural
encodeObs ns =
  let idxs = map round (Torch.asValue ns :: [Float]) :: [Int]
      oneHot = Torch.asTensor [[if i == k then 1.0 else 0.0 :: Float | k <- [0 .. 18]] | i <- idxs]
   in encodeBatch [0 .. 18] oneHot
