{-# LANGUAGE TypeApplications #-}

-- | Grammatical layer (D) — INTERPRETATION for MNIST-addition: the ONE 'mnistSentence'
--   (D/Signature) read in BOTH monads, side by side. One formula, two interpretations -- they
--   differ ONLY along the iteration-vs-vectorization axis (the @Guard@ + the observation @eta n@):
--
--                      the guard (the data)         the observed sum n = @eta n@
--     Dist   (DATA):   a list  @[(x,y,n)]@ -> fold   @pure n@          (a point mass, per element)
--     LogVec (TENS):   one batch @(xs,ys,ns)@-> mean  @encode@(one-hot) (the batched @eta@, a leaf)
--
--   In BOTH, @n@ is the certain observation as a monadic value (@eta n@), bound with @s <- n@; the
--   @LogVec@ realization of @eta@ for a batch of certain sums IS the one-hot leaf (the encode). The
--   @Dist@ reading is the (non-differentiable) probability reading; the @LogVec@ reading is the 'sat'.
module MnistAddition.D_Grammatical.Interpretation
  ( mnistAxiomData,
    mnistAxiomTens,
  )
where

import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogVec (LogVec)
import B_Logical.Interpretations.Boolean () -- A2MonBLat Dist Bool   (the fold quantifier)
import B_Logical.Interpretations.TensorBool () -- A2MonBLat LogVec Bool (the batched quantifier)
import MnistAddition.C_Domain.Interpretation () -- the MnistKlRel instances
import MnistAddition.C_Domain.Signature (Image, Natural) -- the sorts
import MnistAddition.D_Grammatical.Signature (mnistSentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | The @Dist@ reading (probability; not differentiable): ITERATE the dataset of triples; the
--   observed sum enters as @pure n@ (= @eta n@). (Not called at runtime -- the parallel to
--   'mnistAxiomTens'; the report reads per-pair via 'mnistFormula' directly.)
mnistAxiomData :: Weights -> [(Image, Image, Natural)] -> Dist Bool
mnistAxiomData theta dataset =
  mnistSentence @Dist () [(x, y, pure n) | (x, y, n) <- dataset] theta

-- | The @LogVec@ reading (differentiable training, the 'sat'): a pure pass-through. The batch
--   ALREADY carries the observed sum as @eta n@ (a @LogVec Natural@ leaf, built by the E layer),
--   so there is no lifting here -- just read the sentence at @\@LogVec@ over the batch.
mnistAxiomTens :: Weights -> (Image, Image, LogVec Natural) -> LogVec Bool
mnistAxiomTens theta batch = mnistSentence @LogVec () batch theta
