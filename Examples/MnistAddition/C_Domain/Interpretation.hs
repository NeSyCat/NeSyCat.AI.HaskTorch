{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}

-- | Interpretation I_gamma for MNIST single-digit addition. The sorts are monad-invariant plain
--   types (see "MnistAddition.C_Domain.Signature"), so there is no per-monad sort assignment and
--   no image bridge -- an image is the same tensor in both readings. The only monad-dependent
--   content is 'digit':
--
--     digit \@LogVec = LogLeaf [0..9] . cnn theta   -- raw logits as a LogVec leaf
--     digit \@Dist   = decode . digit \@LogVec theta -- the Dist reading is decode of that leaf
module MnistAddition.C_Domain.Interpretation
  ( module MnistAddition.C_Domain.Signature,
    Params,
    initParams,
  )
where

import A_Categorical.Monads.Bridge (decode)
import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogVec (LogVec (..))
import MnistAddition.C_Domain.Signature (Digit, Image, MnistKlRel (..))
import C_Domain.NeuralNets.MnistCNN (cnn, cnnArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)

-- | The parameter space: the pure weights of 'cnnArch'.
type Params = Weights

-- | Draw the initial theta_0 — fresh weights for 'cnnArch'.
initParams :: IO Params
initParams = sampleWeights cnnArch

instance MnistKlRel LogVec where
  digit :: Weights -> Image -> LogVec Digit
  digit theta img = LogLeaf [0 .. 9] (cnn theta img) -- raw logits over 0..9 (no softmax)

instance MnistKlRel Dist where
  digit :: Weights -> Image -> Dist Digit
  digit theta = decode . digit @LogVec theta
    -- the Dist reading IS 'decode' of the LogVec leaf (an image is the same tensor; no bridge)
