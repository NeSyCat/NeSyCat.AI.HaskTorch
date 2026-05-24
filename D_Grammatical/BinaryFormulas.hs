{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Abstract binary classification formula.
--   Uses bigWedge from LogicalQuantSignature for quantification.
--   Works for any universe (GeomU, MeasU, etc.).
module D_Grammatical.BinaryFormulas
  ( binaryPredicate,
    binarySentence,
  )
where

import A_Categorical.CategoricalSignature (Universe (..))
import B_Logical.LogicalQuantSignature (LogicalQuantSignature (..), Guard)
import B_Logical.LogicalSignature (LogicalSignature (..))
import C_Domain.BinarySignature (BinaryRel (..), BinaryKlRel (..), BinarySorts (..))
import C_Domain.Models.MLP (ParamsMLP)

-- | Abstract pointwise predicate for binary classification.
binaryPredicate ::
  forall u.
  ( BinaryKlRel u,
    LogicalSignature u (Omega u),
    Monad (M u)
  ) =>
  ParamsLogic (Omega u) ->
  ParamsMLP ->
  Point u ->
  M u (Omega u)
binaryPredicate lp paramMLP pt = do
  pred <- classifierA @u paramMLP pt
  let label = labelA @u pt
  let and = wedge lp
  let imply = implies lp
  return ((label `imply` pred) `and` (neg label `imply` neg pred))

-- | Sentence: forall x in S. phi(x) -- a guarded quantifier.
--   The guard (Guard u a) specifies the subset S to quantify over.
--   The predicate (binaryPredicate) is pointwise on elements of type a.
binarySentence ::
  forall u a.
  ( BinaryKlRel u,
    LogicalSignature u (Omega u),
    LogicalQuantSignature a u (Omega u),
    Monad (M u),
    a ~ Point u
  ) =>
  ParamsLogic (Omega u) ->
  Guard u a ->
  ParamsMLP ->
  M u (Omega u)
binarySentence lp guard paramMLP =
  bigWedge lp guard
    (binaryPredicate @u lp paramMLP)
