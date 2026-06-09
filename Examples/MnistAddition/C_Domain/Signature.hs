{-# LANGUAGE FlexibleContexts #-}

-- | Non-logical signature for MNIST single-digit addition.
--
--   The sorts are monad-invariant plain types. The ONLY monad-dependent symbol is the neural
--   Kleisli function 'digit' (its monad is @Dist@ or @LogVec@); @(+)@ and @(==)@ are plain Prelude
--   functions used directly in the formula. The observed sum @n@
--   enters the formula as a CERTAIN monadic value @m Natural@ (= @eta n@: @pure n@ in @Dist@, the
--   batched one-hot leaf in @LogVec@), bound like the digits -- so @(+)@ and @(=)@ are host ops
--   on three bound values, and the marginalization (the @Sigma@) is the monad's bind.
module MnistAddition.C_Domain.Signature
  ( Image,
    Digit,
    Natural,
    Omega,
    MnistKlFun (..),
  )
where

import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (monad-invariant plain types).
type Image = Torch.Tensor -- a (batch of) image(s) [.,1,28,28]
type Digit = Int          -- a digit index 0..9
type Natural = Int        -- a sum index 0..18
type Omega = Bool         -- the truth object

-- | The neural digit classifier: the ONE monad-dependent symbol (its distribution monad @m@ is
--   @Dist@ for the probability reading, @LogVec@ for the differentiable one).
class (Monad m) => MnistKlFun m where
  digit :: Weights -> Image -> m Digit
