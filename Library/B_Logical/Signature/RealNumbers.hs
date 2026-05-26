{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | SIGNATURE of (extended) real arithmetic -- the carrier \bar{\mathbb{R}} =
--   \mathbb{R} \cup \{-\infty, +\infty\}. NOT a \top/\bot/\wedge/\vee lattice: the
--   bounds are the two infinities and the operations are the ordered-field ones
--   (+, -, \times, \div), PLUS the aggregations \sum and \prod (the "quant" symbols,
--   parallel to the logical \bigoplus / \bigotimes). One vocabulary; an interpretation
--   (e.g. @Double@, or a @Torch.Tensor@) is an instance defining all of it.
module B_Logical.Signature.RealNumbers (RealNumbersSignature (..)) where

import A_Categorical.CategoricalSignature (Universe (..))
import B_Logical.Signature.Guard (Guard)

class (Universe u, Monad (M u)) => RealNumbersSignature u tau | tau -> u where
  -- bounds + field operations
  posInf :: tau -- ^ +\infty  (greatest extended real)
  negInf :: tau -- ^ -\infty  (least extended real)
  plus :: tau -> tau -> tau -- ^ +
  minus :: tau -> tau -> tau -- ^ -  (subtraction)
  times :: tau -> tau -> tau -- ^ \times
  divide :: tau -> tau -> tau -- ^ \div
  -- aggregations
  bigSum :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \sum
  bigProduct :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \prod
