{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Grammatical layer (D) — SIGNATURE for the Binary example: the abstract binary
--   classification formula (the axiom), universe-polymorphic over @u@. Uses
--   bigWedge from A2MonBLat for quantification; this single formula is
--   interpreted per universe in InterpretationData (MeasU) / InterpretationTens (GeomU).
module Binary.D_Grammatical.Signature
  ( binaryPredicate,
    binarySentence,
  )
where

import A_Categorical.CategoricalSignature (Framework (..))
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import Binary.C_Domain.Signature (BinaryKlRel (..), BinaryRel (..), Omega, Point)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | Abstract pointwise predicate for binary classification.
binaryPredicate ::
  forall u.
  ( BinaryKlRel u,
    TwoMonBLat Omega,
    Monad (M u)
  ) =>
  ParamsLogic Omega ->
  Weights ->
  Point ->
  M u Omega
binaryPredicate lp paramMLP pt = do
  pred <- classifierA @u paramMLP pt
  label <- labelA @u pt   -- the label now flows through the monad (a certain leaf), like MNIST's observed sum
  let and = wedge lp
  let imply = implies lp
  return ((label `imply` pred) `and` (neg label `imply` neg pred))

-- | Sentence: forall x in S. phi(x) -- a guarded quantifier.
--   The guard (Guard u a) specifies the subset S to quantify over.
--   The predicate (binaryPredicate) is pointwise on elements of type a.
binarySentence ::
  forall u a.
  ( BinaryKlRel u,
    TwoMonBLat Omega,
    A2MonBLat a u Omega,
    Monad (M u),
    a ~ Point
  ) =>
  ParamsLogic Omega ->
  Guard u a ->
  Weights ->
  M u Omega
binarySentence lp guard paramMLP =
  -- @u@/@Omega@ pinned explicitly: the truth algebra is universe-free.
  bigWedge @a @u @Omega lp guard
    (binaryPredicate @u lp paramMLP)
