{-# LANGUAGE TypeApplications #-}

-- | Grammatical layer (D) — INTERPRETATION for MNIST-addition: the ONE 'mnistSentence'
--   (D/Signature) read in BOTH monads, side by side. One formula, two interpretations -- they
--   differ ONLY along the iteration-vs-vectorization axis (the @Guard@ + the observation):
--
--                      the guard (the data)         the observed sum n
--     Dist   (DATA):   a list  @[(x,y,n)]@ -> fold   @pure n@           (a point mass, per element)
--     LogVec (TENS):   one batch @(xs,ys,ns)@-> mean  @encode@(one-hot) (a batched delta leaf)
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

import A_Categorical.Monads.Bridge (encode)
import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogVec (LogVec)
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
--   sums @ns@ (a @[B]@ tensor of indices) are lifted into @LogVec@ as a CERTAIN batched delta --
--   one-hot over @[0..18]@, then 'encode'. This is @Dist@'s @pure n@ done all-at-once for the
--   batch (the same one-hot-then-'encode' Binary's 'labelA' does for its circle test).
mnistAxiomTens :: Weights -> (Image, Image, Torch.Tensor) -> LogVec Bool
mnistAxiomTens theta (xs, ys, ns) =
  mnistSentence @LogVec () (xs, ys, encodeObs ns) theta
  where
    encodeObs :: Torch.Tensor -> LogVec Natural
    encodeObs idxsTensor =
      let idxs = map round (Torch.asValue idxsTensor :: [Float]) :: [Int]
          oneHot = Torch.asTensor [[if i == k then 1.0 else 0.0 :: Float | k <- [0 .. 18]] | i <- idxs]
       in encode [0 .. 18] oneHot
