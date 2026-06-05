{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Grammatical layer (D) — SIGNATURE for the MNIST example: the single abstract
--   MNIST-addition formula, monad-polymorphic over @m@, INCLUDING its quantifier.
--
--     forall (x,y,n) in data.  n = digit(x) + digit(y)
--
--   Faithful to the theory (presi.tex): @digit : (m)Image -> (m)Digit@, the observed sum enters
--   as @eta n :: m Natural@ (bound with @s <- n@, exactly like the digits), and @(+)@/@(=)@ are
--   plain host functions on the bound values. Written ONCE; only the monad @m@ changes -- @Dist@
--   (probability) or @LogVec@ (differentiable). Interpreted per monad in
--   "MnistAddition.D_Grammatical.Interpretation".
module MnistAddition.D_Grammatical.Signature
  ( mnistFormula,
    mnistSentence,
  )
where

import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import MnistAddition.C_Domain.Signature (Image, MnistKlRel (..), Natural, Omega, eqNat, plus)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | The per-pair FORMULA  @n = digit(x) + digit(y)@, monad-polymorphic over @m@. The observed
--   sum @n :: m Natural@ is bound (@s <- n@) exactly like the digits; @(.+)@/@(.=)@ are plain
--   host ops on the three bound values; the bind supplies the marginalization (the law of total
--   probability for @Dist@, the log-space convolution for @LogVec@).
mnistFormula ::
  forall m.
  (MnistKlRel m) =>
  Weights ->
  (Image, Image, m Natural) ->
  m Omega
mnistFormula theta (x, y, n) =
  let
    (.+) = plus
    (.=) = eqNat
    dig = digit @m theta
   in
    do
      d1 <- dig x
      d2 <- dig y
      s <- n
      return (s .= (d1 .+ d2))

-- | The SENTENCE  @forall (x,y,n) in data. n = digit(x) + digit(y)@ — the whole quantified axiom,
--   abstract over @m@. The guard @Guard m (Image, Image, m Natural)@ is the data quantified over
--   (a list of triples in @Dist@, the batched triple in @LogVec@); the universal is 'bigWedge'.
mnistSentence ::
  forall m.
  ( MnistKlRel m,
    TwoMonBLat Omega,
    A2MonBLat m Omega,
    Monad m
  ) =>
  ParamsLogic Omega ->
  Guard m (Image, Image, m Natural) ->
  Weights ->
  m Omega
mnistSentence lp guard theta =
  bigWedge lp guard (mnistFormula @m theta)
