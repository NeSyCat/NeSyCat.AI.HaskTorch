{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The single abstract MNIST-addition formula, interpreted in any universe.
--
--   One syntax:   forall x y.  add(x,y) = digit(x) + digit(y)
--
--   The per-pair atom below is its body, built only from the interpreted symbols
--   @digit@, @(+)@ (= 'plus') and @(=)@ (= 'eqNat'). There is no existential: the
--   marginalization (the @Sigma@ of the law of total probability) is part of
--   @plus@'s interpretation -- the @Dist@ bind in MeasU, a LogSumExp on logits in
--   GeomU. The universal quantifier is the logic's domain-independent @bigWedge@,
--   applied per universe in the two interpretation modules (reused, not redefined).
module D_Grammatical.Examples.MnistAddition.Formulas
  ( mnistFormula,
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import C_Domain.Examples.MnistAddition.Signature (MnistArith (..), MnistKlRel (..), MnistSorts (..))

-- | The atom  @add(x,y) = digit(x) + digit(y)@  at the observed sum @n@,
--   polymorphic over the universe @u@. The @do@-block never changes; only the
--   symbols are reinterpreted. In MeasU it marginalizes through the @Dist@ bind
--   (@d1,d2 :: Int@); in GeomU it runs once on a batch (@d1,d2 :: logits@) with
--   @plus@ = log-space convolution.
mnistFormula ::
  forall u.
  (MnistKlRel u, MnistArith u) =>
  ThetaCNN u ->
  (Image u, Image u, Natural u) ->
  M u (Omega u)
mnistFormula theta (x, y, n) = do
  d1 <- digit @u theta x              -- digit(x)
  d2 <- digit @u theta y              -- digit(y)
  return (eqNat @u n (plus @u d1 d2)) -- add(x,y) = digit(x) + digit(y)
