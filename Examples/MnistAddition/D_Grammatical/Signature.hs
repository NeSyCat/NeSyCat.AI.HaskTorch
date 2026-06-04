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
--   then interpreted per universe in InterpretationData (MeasU) / InterpretationTens
--   (GeomU). There is no existential: the marginalization (the @Sigma@ of the law of
--   total probability) is part of @plus@'s interpretation.
module MnistAddition.D_Grammatical.Signature
  ( mnistFormula,
    mnistSentence,
  )
where

import A_Categorical.CategoricalSignature (Framework (..))
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import MnistAddition.C_Domain.Signature (Image, MnistKlRel (..), Natural, Omega, eqNat, plus)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | The per-pair PREDICATE  @add(x,y) = digit(x) + digit(y)@  at the observed sum @n@,
--   polymorphic over the universe @u@. The @do@-block never changes; only the symbols
--   are reinterpreted. In MeasU it marginalizes through the @Dist@ bind (@d1,d2 :: Int@);
--   in GeomU it runs once on a batch (@d1,d2 :: logits@) with @plus@ = log-space convolution.
mnistFormula ::
  forall u.
  (MnistKlRel u) =>
  Weights ->
  (Image, Image, M u Natural) ->
  M u Omega
mnistFormula theta (x, y, n) =
  let
    (.+) = plus
    (.=) = eqNat
    dig = digit @u theta
  in do
    d1 <- dig x
    d2 <- dig y
    s <- n
    return (s .= (d1 .+ d2))

-- | The SENTENCE  @forall (x,y,n) in data. add(x,y) = digit(x) + digit(y)@ — the whole
--   quantified axiom, abstract over @u@. The guard @Guard u a@ is the data quantified over
--   (a list of triples in MeasU, the batched triple in GeomU); the universal is the logic's
--   'bigWedge'. Interpreted per universe by 'mnistAxiomTens' (GeomU) / 'mnistAxiomData' (MeasU).
mnistSentence ::
  forall u a.
  ( MnistKlRel u,
    TwoMonBLat Omega,
    A2MonBLat a u Omega,
    Monad (M u),
    a ~ (Image, Image, M u Natural)
  ) =>
  ParamsLogic Omega ->
  Guard u a ->
  Weights ->
  M u Omega
mnistSentence lp guard theta =
  -- @u@/@Omega@ are pinned explicitly: the truth algebra is universe-free, so the universe is
  -- supplied at the quantifier.
  bigWedge @a @u @Omega lp guard (mnistFormula @u theta)
