{-# LANGUAGE TypeApplications #-}

-- | Interpretation for MNIST multi-digit addition. The sorts are monad-invariant plain types, so
--   'digit' is the only interpreted Kleisli function -- and it is EXACTLY the single-digit interpretation
--   (the same @cnnArch@ classifier, reused per image):
--
--     digit \@LogVec = LogLeaf [0..9] . cnn theta       -- raw logits as a LogVec leaf
--     digit \@Dist   = decode . digit \@LogVec theta     -- the Dist reading is decode of that leaf
--
--   The observed sum @n@ is NOT interpreted here -- it enters the formula as a certain @m Natural@
--   (= @eta n@), provided by the E layer (the @encode@ = batched @eta@).
module MnistMultiDigit.C_Domain.Interpretation
  ( module MnistMultiDigit.C_Domain.Signature,
    Params,
    initParams,
  )
where

import A_Categorical.Monads.Bridge (decode)
import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogVec (LogVec (..))
import MnistMultiDigit.C_Domain.Signature
import C_Domain.NeuralNets.MnistCNN (cnn, cnnArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)

-- | The parameter space: the pure weights of 'cnnArch' (one shared digit CNN for all four images).
type Params = Weights

-- | Draw the initial theta_0 -- fresh weights for 'cnnArch'.
initParams :: IO Params
initParams = sampleWeights cnnArch

instance MnistKlFun LogVec where
  digit theta img = LogLeaf [0 .. 9] (cnn theta img) -- raw logits over 0..9 (no softmax)

instance MnistKlFun Dist where
  digit theta = decode . digit @LogVec theta
