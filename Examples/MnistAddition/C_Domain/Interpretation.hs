{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE InstanceSigs #-}

-- | Interpretation I_gamma for MNIST single-digit addition. The sorts are monad-invariant plain
--   types, so 'digit' is the only interpreted Kleisli function:
--
--     digit \@LogVec theta = (>>= \img -> LogLeaf [0..9] (cnn theta img))   -- Kleisli extension (two-sided)
--     digit \@Dist   theta = decode . digit \@LogVec theta . encDist          -- decode . dig^LogVec . encode (square 2)
--
--   'digit' is now genuinely two-sided  (m)Image -> (m)Digit: the image enters as the certain
--   monadic value @eta x@ (the encode = @pure@), exactly like the observed sum @eta n@, and the bind
--   supplies the lift. On a certain image @eta x@ the left-unit law collapses this to the same
--   @LogLeaf [0..9] (cnn theta x)@ as before, so the computation (and cost) is unchanged.
module MnistAddition.C_Domain.Interpretation
  ( module MnistAddition.C_Domain.Signature,
    Params,
    initParams,
  )
where

import A_Categorical.Monads.Bridge (decode, encDist)
import A_Categorical.Monads.Dist (Dist)
import A_Categorical.Monads.LogVec (LogVec (..))
import MnistAddition.C_Domain.Signature
import C_Domain.NeuralNets.MnistCNN (cnn, cnnArch)
import C_Domain.NeuralNets.DSL.Semantics (Weights, sampleWeights)

-- | The parameter space: the pure weights of 'cnnArch'.
type Params = Weights

-- | Draw the initial theta_0 -- fresh weights for 'cnnArch'.
initParams :: IO Params
initParams = sampleWeights cnnArch

instance MnistKlFun LogVec where
  digit :: Weights -> LogVec Image -> LogVec Digit
  digit theta = (>>= (LogLeaf [0 .. 9] . cnn theta))
                                  
instance MnistKlFun Dist where
  digit :: Weights -> Dist Image -> Dist Digit
  digit theta = decode . digit @LogVec theta . encDist
