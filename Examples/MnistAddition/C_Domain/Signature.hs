{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

-- | Non-logical signature for MNIST single-digit addition.
--
--   The sorts are monad-INVARIANT, so they are plain types -- NOT @... u@ associated types.
--   An image is a tensor, a digit/sum an @Int@ index, the truth a @Bool@: the per-value
--   distribution lives in the monad @m@, not in the sort. The ONLY symbol that depends on the
--   monad is 'digit' (its monad is @Dist@ or @LogVec@); @(+)@ and @(=)@ are monad-free host
--   operations (the @Sigma@ / marginalization lives in the monad's bind).
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

-- | The neural digit classifier: the ONE monad-dependent symbol (its distribution monad @m@ is
--   @Dist@ for the probability reading, @LogVec@ for the differentiable one).
class (Monad m) => MnistKlRel m where
  digit :: Weights -> Image -> m Digit
