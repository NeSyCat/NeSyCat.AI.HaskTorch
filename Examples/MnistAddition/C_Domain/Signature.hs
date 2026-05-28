{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | Non-logical signature for MNIST single-digit addition (survey Ex. 2):
--
--     Sorts : Image, Digit, Natural
--     Fun   : digit : Image -> Digit   (the neural classifier, a Kleisli relation)
--             add   : Image^2 -> Natural   (the observed sum, supplied by the data)
--             (+)   : Digit^2 -> Natural   (ordinary addition, host-level)
--     Rel   : (=)   : Natural^2
--
--   Here we declare the sorts and the only learned symbol, @digit@. The pure
--   symbols @(+)@ and @(=)@ are interpreted in the grammatical layer (over the
--   chosen monad), so the axiom  forall x y. add(x,y) = digit(x) + digit(y)  is
--   assembled compositionally, never declared.
module MnistAddition.C_Domain.Signature
  ( MnistSorts (..),
    MnistParams (..),
    MnistKlRel (..),
    MnistArith (..),
    MnistBridge (..),
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import Data.Kind (Type)

-- | Sort symbols, assigned concrete objects by an interpretation.
class (Universe u) => MnistSorts u where
  type Image u :: Type
  type Digit u :: Type
  type Natural u :: Type
  type Omega u :: Type

-- ============================================================
--  Parameter spaces (horizontal sorts): the actor object ThetaCNN where the CNN's
--  learnable parameters live (the vertical sorts above hold the variables).
-- ============================================================

-- | Parameter-space symbol (horizontal sort), assigned a concrete space (the CNN
--   weight space) by an interpretation.
class (Universe u) => MnistParams u where
  type ThetaCNN u :: Type

-- | The neural digit classifier as a parametrized Kleisli relation
--   @digit : Theta . Image -> M Digit@. Its semantics is the CNN (logits) bridged
--   into a distribution over digits by the interpretation.
class (MnistSorts u, MnistParams u, Monad (M u)) => MnistKlRel u where
  digit :: ThetaCNN u -> Image u -> M u (Digit u)

-- | Domain arithmetic/relation symbols of the MNIST vocabulary:
--     @(+) : Digit^2 -> Natural@   (the @plus@ symbol)
--     @(=) : Natural^2@            (the @eqNat@ relation symbol)
--   Interpreted per universe. The marginalization (the @Sigma@ of the law of
--   total probability) lives in @plus@'s interpretation: a host @+@ under the
--   @Dist@ bind in MeasU, a LogSumExp over @d1+d2=s@ on logits in GeomU. The
--   formula therefore stays existential-free and identical across both.
class (MnistSorts u) => MnistArith u where
  plus :: Digit u -> Digit u -> Natural u
  eqNat :: Natural u -> Natural u -> Omega u

-- | Bridge between two universe interpretations: encode an image into the other
--   universe (mirrors the binary @encPoint@). (Digit decoding is no longer a bridge:
--   each universe's @digit@ builds its own distribution from the CNN logits.)
class (MnistSorts from, MnistSorts to) => MnistBridge from to where
  encImage :: Image from -> Image to
