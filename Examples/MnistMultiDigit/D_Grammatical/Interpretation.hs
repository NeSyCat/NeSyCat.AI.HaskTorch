{-# LANGUAGE TypeApplications #-}

-- | Grammatical layer (D) — INTERPRETATION for MNIST multi-digit addition: the ONE 'multiSentence'
--   (D/Signature) read in BOTH monads, side by side. One formula, two interpretations -- they
--   differ ONLY along the iteration-vs-vectorization axis (the @Guard@ + the observation @eta n@).
--   The @Dist@ reading is the (non-differentiable) probability reading; the @LogTens@ reading is the
--   'sat'. The @LogTens@ reading is a pure pass-through: the batch ALREADY carries the observed sum
--   as @eta n@ (a @LogTens Natural@ leaf over [0..198], built by the E layer), so there is no lifting
--   here.
module MnistMultiDigit.D_Grammatical.Interpretation
  ( multiAxiomData,
    multiAxiomTens,
  )
where

import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogTens (LogTens)
import B_Logical.Interpretations.Boolean () -- A2MonBLat Dist Bool   (the fold quantifier)
import B_Logical.Interpretations.TensorBool () -- A2MonBLat LogTens Bool (the convolution quantifier)
import MnistMultiDigit.C_Domain.Interpretation () -- the MnistKlFun instances
import MnistMultiDigit.C_Domain.Signature (Image, Natural)
import MnistMultiDigit.D_Grammatical.Signature (multiSentence)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | The @Dist@ reading (probability; not differentiable): ITERATE the dataset of 5-tuples; the
--   observed sum enters as @pure n@ (= @eta n@). (Not called at runtime; the parallel to
--   'multiAxiomTens'.)
multiAxiomData :: Weights -> [(Image, Image, Image, Image, Natural)] -> Dist Bool
multiAxiomData theta dataset =
  multiSentence @Dist () [(a, b, c, d, pure n) | (a, b, c, d, n) <- dataset] theta

-- | The @LogTens@ reading (differentiable training, the 'sat'): a pure pass-through over the batch,
--   which already carries the observed sum as @eta n@ (a @LogTens Natural@ leaf, built by the E
--   layer). The marginalization over the four unknown digits is the log-space convolution (the
--   bind), so no @O(10^4 * 199)@ joint is ever formed.
multiAxiomTens :: Weights -> (Image, Image, Image, Image, LogTens Natural) -> LogTens Bool
multiAxiomTens theta batch = multiSentence @LogTens () batch theta
