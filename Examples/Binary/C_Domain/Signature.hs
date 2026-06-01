{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module Binary.C_Domain.Signature where

import A_Categorical.CategoricalSignature (Framework (..))
import Data.Kind (Type)

-- | Non-logical signature Sigma_gamma for the Binary Classification domain.
--
--   Vertical points (sorts)        : Point, Omega   -- objects of the domain category C_gamma
--   Horizontal point (param space) : Theta          -- object of the actor A
--   Relation symbols               : labelA      : Point -> Omega           (Tarski)
--                                     classifierA : Theta . Point -> M Omega (parametrized Kleisli)
--
--   Theta is only a *symbol* here; its semantics (e.g. an MLP weight space, a
--   morphism of the actor) is supplied by an interpretation. The monad is M u.

-- | Sort symbols, assigned to concrete objects by an interpretation.
class (Framework u) => BinarySorts u where
  type Point u :: Type
  type Omega u :: Type

-- ============================================================
--  Parameter spaces (horizontal sorts): the actor objects Theta where the
--  learnable parameters live. The vertical sorts above hold the variables (x, y);
--  these hold the parameters theta. One symbol per network -- here a single Theta.
-- ============================================================

-- | Parameter-space symbols (horizontal sorts), each assigned a concrete space
--   (e.g. an MLP weight space) by an interpretation.
class (Framework u) => BinaryParams u where
  type Theta u :: Type

-- | Tarski relation symbol.
class (BinarySorts u) => BinaryRel u where
  labelA :: Point u -> Omega u

-- | Parametrized Kleisli relation symbol. @Theta u@ is the parameter-space
--   symbol (a horizontal point); its semantics is fixed by the interpretation.
class (BinaryRel u, BinaryParams u, Monad (M u)) => BinaryKlRel u where
  classifierA :: Theta u -> Point u -> M u (Omega u)

-- | Bridge for encoding/decoding between two universe interpretations.
class (BinarySorts from, BinarySorts to) => BinaryBridge from to where
  encPoint :: Point from -> Point to
  decOmega :: Omega to -> M from (Omega from)
