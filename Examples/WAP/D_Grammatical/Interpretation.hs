{-# LANGUAGE TypeApplications #-}

-- | Grammatical layer (D) -- INTERPRETATION for WAP: the ONE 'wapSentence' (D/Signature)
--   read in BOTH monads, side by side. One formula, two interpretations -- they differ ONLY
--   along the iteration-vs-vectorization axis (the @Guard@ + the observation @\eta (ns, y)@):
--
--                      the guard (the data)            the observation @\eta (ns, y)@
--     Dist   (DATA):   a list  @[(s, obs)]@ -> fold    @pure (ns, y)@   (a point mass, per item)
--     LogTens (TENS):   one batch pair      -> mean     @encode@ (one-hot leaf over the batch's
--                                                       distinct pairs -- built by the E layer)
--
--   In BOTH, the observation is the certain pair as a monadic value (@\eta (ns, y)@), bound
--   @(ns, y) <- obs@ exactly like the sketch decisions; the @LogTens@ realization of @\eta@
--   for a batch is the one-hot leaf (the E layer's @encode@), so this interpretation is a
--   pure pass-through. The @Dist@ reading is the (non-differentiable) probability reading;
--   the @LogTens@ reading is the 'sat'.
module WAP.D_Grammatical.Interpretation
  ( wapAxiomData,
    wapAxiomTens,
  )
where

import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogTens (LogTens)
import B_Logical.Interpretations.Boolean () -- A2MonBLat Dist Bool   (the fold quantifier)
import B_Logical.Interpretations.TensorBool () -- A2MonBLat LogTens Bool (the batched quantifier)
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import WAP.C_Domain.Interpretation () -- the WapKlFun instances
import WAP.C_Domain.Signature (Answer, Numbers, Problem)
import WAP.D_Grammatical.Signature (wapSentence)

-- | The @Dist@ reading (probability; not differentiable): ITERATE the dataset of items; each
--   problem a singleton batch, the observation as @pure (ns, y)@ (= @\eta (ns, y)@). (Not
--   called at runtime -- the parallel to 'wapAxiomTens'.)
wapAxiomData :: Weights -> [(Problem, Numbers, Answer)] -> Dist Bool
wapAxiomData theta ds =
  wapSentence @Dist () [([p], pure (ns, y)) | (p, ns, y) <- ds] theta

-- | The @LogTens@ reading (differentiable training, the 'sat'): a pure pass-through. The
--   batch ALREADY carries the observation as @\eta (ns, y)@ (a @LogTens (Numbers, Answer)@
--   leaf over the batch's distinct pairs, built by the E layer), so there is no lifting
--   here -- just read the sentence at @\@LogTens@ over the batch.
wapAxiomTens :: Weights -> ([Problem], LogTens (Numbers, Answer)) -> LogTens Bool
wapAxiomTens theta batch = wapSentence @LogTens () batch theta
