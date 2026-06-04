{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

-- | Non-logical signature for MNIST single-digit addition.
--
--   The sorts are universe-INVARIANT, so they are plain types -- NOT @... u@ associated types.
--   An image is a tensor, a digit/sum an @Int@ index, the truth a @Bool@: the per-value
--   distribution lives in the monad @M u@, not in the sort. The ONLY symbol that depends on the
--   universe is 'digit' (its monad is @Dist@ in MeasU, @LogVec@ in GeomU); @(+)@ and @(=)@ are
--   universe-free host operations (the @Sigma@ / marginalization lives in the monad's bind).
module MnistAddition.C_Domain.Signature
  ( Image,
    Digit,
    Natural,
    Omega,
    plus,
    eqNat,
    MnistKlRel (..),
  )
where

import A_Categorical.CategoricalSignature (Framework (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (universe-invariant plain types).
type Image = Torch.Tensor -- a (batch of) image(s) [.,1,28,28]
type Digit = Int          -- a digit index 0..9
type Natural = Int        -- a sum index 0..18
type Omega = Bool         -- the truth object

-- | @(+) : Digit^2 -> Natural@, @(=) : Natural^2@ -- universe-free host operations.
plus :: Digit -> Digit -> Natural
plus = (+)

eqNat :: Natural -> Natural -> Omega
eqNat = (==)

-- | The neural digit classifier: the ONE universe-dependent symbol (its distribution monad
--   @M u@ is @Dist@ in MeasU, @LogVec@ in GeomU).
class (Framework u, Monad (M u)) => MnistKlRel u where
  digit :: Weights -> Image -> M u Digit
