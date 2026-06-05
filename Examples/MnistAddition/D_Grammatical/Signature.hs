{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Grammatical layer (D) — SIGNATURE for the MNIST example: the single abstract
--   MNIST-addition formula, universe-polymorphic over @u@, INCLUDING its quantifier.
--   Like Binary's @binarySentence@, the whole sentence
--
--     forall (x,y,n) in data.  add(x,y) = digit(x) + digit(y)
--
--   is written ONCE here (the predicate 'mnistFormula' under the logic's 'bigWedge'),
--   then interpreted per monad in InterpretationData (@Dist@) / InterpretationTens
--   (@LogVec@). There is no existential: the marginalization (the @Sigma@ of the law of
--   total probability) is part of @plus@'s interpretation.
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

-- | The per-pair PREDICATE  @add(x,y) = digit(x) + digit(y)@  at the observed sum @n@,
--   polymorphic over the monad @m@. The @do@-block never changes; only the symbols
--   are reinterpreted. In @Dist@ it marginalizes through the bind (@d1,d2 :: Int@);
--   in @LogVec@ it runs once on a batch (@d1,d2 :: logits@) with @plus@ = log-space convolution.
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
  in do
    d1 <- dig x
    d2 <- dig y
    s <- n
    return (s .= (d1 .+ d2))

-- | The SENTENCE  @forall (x,y,n) in data. add(x,y) = digit(x) + digit(y)@ — the whole
--   quantified axiom, abstract over @m@. The guard @Guard m a@ is the data quantified over
--   (a list of triples in @Dist@, the batched triple in @LogVec@); the universal is the logic's
--   'bigWedge'. Interpreted per monad by 'mnistAxiomTens' (@LogVec@) / 'mnistAxiomData' (@Dist@).
mnistSentence ::
  forall m a.
  ( MnistKlRel m,
    TwoMonBLat Omega,
    A2MonBLat a m Omega,
    Monad m,
    a ~ (Image, Image, m Natural)
  ) =>
  ParamsLogic Omega ->
  Guard m a ->
  Weights ->
  m Omega
mnistSentence lp guard theta =
  -- @m@/@Omega@ are pinned explicitly: the truth algebra is monad-free, so the monad is
  -- supplied at the quantifier.
  bigWedge @a @m @Omega lp guard (mnistFormula @m theta)
