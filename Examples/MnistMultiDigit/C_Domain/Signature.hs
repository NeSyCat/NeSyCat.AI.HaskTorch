{-# LANGUAGE FlexibleContexts #-}

-- | Non-logical signature for MNIST MULTI-digit addition (two TWO-digit numbers).
--
--   The sorts are monad-invariant plain types. The ONLY monad-dependent symbol is the neural
--   Kleisli function 'digit' (its monad is @Dist@ or @LogTens@) -- the SAME single-digit classifier, called
--   four times (one per image). @number@ (@10*hi + lo@), @(+)@ ('plus') and @(=)@ ('eqNat') are
--   plain host functions, applied to the bound digit values inside the formula. The observed sum
--   @n@ enters the formula as a CERTAIN monadic value @m Natural@ (= @eta n@), bound like the
--   digits -- so the marginalization (the @Sigma@) is the monad's bind, realized as the log-space
--   convolution of the four digit leaves (variable elimination; no @O(10^4 * 199)@ joint).
module MnistMultiDigit.C_Domain.Signature
  ( Image,
    Digit,
    Natural,
    Omega,
    number,
    plus,
    eqNat,
    MnistKlFun (..),
  )
where

import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (monad-invariant plain types).
type Image = Torch.Tensor -- a (batch of) image(s) [.,1,28,28]
type Digit = Int          -- a digit index 0..9
type Natural = Int        -- a sum index 0..198 (two two-digit numbers)
type Omega = Bool         -- the truth object

-- | @number hi lo = 10*hi + lo@ -- compose a two-digit number from its digits (plain host fn).
number :: Digit -> Digit -> Natural
number hi lo = 10 * hi + lo

-- | @(+) : Natural^2 -> Natural@, @(=) : Natural^2 -> Omega@ -- plain host functions on the bound
--   (host) values inside the formula.
plus :: Natural -> Natural -> Natural
plus = (+)

eqNat :: Natural -> Natural -> Omega
eqNat = (==)

-- | The neural digit classifier: the ONE monad-dependent symbol (its distribution monad @m@ is
--   @Dist@ for the probability reading, @LogTens@ for the differentiable one) -- the SAME Kleisli function
--   as single-digit addition, reused here for all four images.
class (Monad m) => MnistKlFun m where
  digit :: Weights -> Image -> m Digit
