{-# LANGUAGE FlexibleContexts #-}

-- | Non-logical signature for MNIST single-digit addition.
--
--   The sorts are monad-invariant plain types. The ONLY monad-dependent symbol is the neural
--   relation 'digit' (its monad is @Dist@ or @LogVec@); @(+)@ ('plus') and @(=)@ ('eqNat') are
--   plain host functions, applied to the bound values inside the formula. The observed sum @n@
--   enters the formula as a CERTAIN monadic value @m Natural@ (= @eta n@: @pure n@ in @Dist@, the
--   batched one-hot leaf in @LogVec@), bound like the digits -- so @(+)@ and @(=)@ are host ops
--   on three bound values, and the marginalization (the @Sigma@) is the monad's bind.
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

import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (monad-invariant plain types).
type Image = Torch.Tensor -- a (batch of) image(s) [.,1,28,28]
type Digit = Int          -- a digit index 0..9
type Natural = Int        -- a sum index 0..18
type Omega = Bool         -- the truth object

-- | @(+) : Digit^2 -> Natural@, @(=) : Natural^2 -> Omega@ -- plain host functions, applied to the
--   bound (host) values inside the formula.
plus :: Digit -> Digit -> Natural
plus = (+)

eqNat :: Natural -> Natural -> Omega
eqNat = (==)

-- | The neural digit classifier: the ONE monad-dependent symbol (its distribution monad @m@ is
--   @Dist@ for the probability reading, @LogVec@ for the differentiable one).
class (Monad m) => MnistKlRel m where
  digit :: Weights -> Image -> m Digit
