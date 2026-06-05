{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Grammatical layer (D) — SIGNATURE for the MNIST multi-digit example: the single abstract
--   formula, monad-polymorphic over @m@, INCLUDING its quantifier.
--
--     forall (x1,x2,y1,y2,n) in data.  n = number(x1,x2) + number(y1,y2)
--                                         = (10*digit(x1)+digit(x2)) + (10*digit(y1)+digit(y2))
--
--   The four digits are four calls of the SAME 'digit' relation; the observed sum enters as
--   @eta n :: m Natural@ (bound @s <- n@, exactly like the digits); @number@/@(+)@/@(=)@ are
--   plain host functions on the bound values. Written ONCE; only the monad @m@ changes.
--   Interpreted per monad in "MnistMultiDigit.D_Grammatical.Interpretation".
module MnistMultiDigit.D_Grammatical.Signature
  ( multiFormula,
    multiSentence,
  )
where

import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import MnistMultiDigit.C_Domain.Signature (Image, MnistKlRel (..), Natural, Omega, eqNat, number, plus)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | The per-example FORMULA  @n = number(x1,x2) + number(y1,y2)@, monad-polymorphic over @m@. The
--   observed sum @n :: m Natural@ is bound (@s <- n@) exactly like the digits; @number@/@(.+)@/@(.=)@
--   are plain host ops on the five bound values; the bind supplies the marginalization (the law of
--   total probability for @Dist@; the log-space convolution of the four digit leaves for @LogVec@).
multiFormula ::
  forall m.
  (MnistKlRel m) =>
  Weights ->
  (Image, Image, Image, Image, m Natural) ->
  m Omega
multiFormula theta (x1, x2, y1, y2, n) =
  let (.+) = plus
      (.=) = eqNat
      dig = digit @m theta
   in do
        d1 <- dig x1 -- high digit of number A
        d2 <- dig x2 -- low  digit of number A
        d3 <- dig y1 -- high digit of number B
        d4 <- dig y2 -- low  digit of number B
        s <- n
        return (s .= (number d1 d2 .+ number d3 d4))

-- | The SENTENCE  @forall (x1,x2,y1,y2,n) in data. n = number(x1,x2) + number(y1,y2)@ — the whole
--   quantified axiom, abstract over @m@. The guard is the data quantified over; the universal is
--   'bigWedge'.
multiSentence ::
  forall m.
  ( MnistKlRel m,
    TwoMonBLat Omega,
    A2MonBLat m Omega,
    Monad m
  ) =>
  ParamsLogic Omega ->
  Guard m (Image, Image, Image, Image, m Natural) ->
  Weights ->
  m Omega
multiSentence lp guard theta =
  bigWedge lp guard (multiFormula @m theta)
