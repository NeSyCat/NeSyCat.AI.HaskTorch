{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Grammatical layer (D) — SIGNATURE for the Binary example: the abstract binary
--   classification formula (the axiom), monad-polymorphic over @m@. Uses
--   bigWedge from A2MonBLat for quantification; this single formula is
--   read in both monads in 'Binary.D_Grammatical.Interpretation' (@Dist@ / @LogVec@).
module Binary.D_Grammatical.Signature
  ( binaryPredicate,
    binarySentence,
  )
where

import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import Binary.C_Domain.Signature (BinaryKlRel (..), BinaryRel (..), Omega, Point)
import C_Domain.NeuralNets.DSL.Semantics (Weights)

-- | Abstract pointwise predicate for binary classification.
binaryPredicate ::
  forall m.
  ( BinaryKlRel m,
    TwoMonBLat Omega,
    Monad m
  ) =>
  ParamsLogic Omega ->
  Weights ->
  Point ->
  m Omega
binaryPredicate lp paramMLP pt = do
  pred <- classifierA @m paramMLP pt
  label <- labelA @m pt   -- the label now flows through the monad (a certain leaf), like MNIST's observed sum
  let and = wedge lp
  let imply = implies lp
  return ((label `imply` pred) `and` (neg label `imply` neg pred))

-- | Sentence: forall x in S. phi(x) -- a guarded quantifier.
--   The guard (Guard m a) specifies the subset S to quantify over.
--   The predicate (binaryPredicate) is pointwise on elements of type a.
binarySentence ::
  forall m.
  ( BinaryKlRel m,
    TwoMonBLat Omega,
    A2MonBLat m Omega,
    Monad m
  ) =>
  ParamsLogic Omega ->
  Guard m Point ->
  Weights ->
  m Omega
binarySentence lp guard paramMLP =
  -- the point type is inferred from the predicate; only the monad @m@ is supplied (by the caller).
  bigWedge lp guard (binaryPredicate @m lp paramMLP)
