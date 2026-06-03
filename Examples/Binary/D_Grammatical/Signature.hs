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
import Binary.C_Domain.Signature (BinaryRel (..), BinaryKlRel (..), BinaryParams (..), BinarySorts (..))

-- | Abstract pointwise predicate for binary classification.
binaryPredicate ::
  forall u.
  ( BinaryKlRel u,
    TwoMonBLat (Omega u),
    Monad (M u)
  ) =>
  ParamsLogic (Omega u) ->
  Theta u ->
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
    TwoMonBLat (Omega u),
    A2MonBLat a u (Omega u),
    Monad (M u),
    a ~ Point u
  ) =>
  ParamsLogic (Omega u) ->
  Guard u a ->
  Theta u ->
  M u (Omega u)
binarySentence lp guard paramMLP =
  -- @u@/@(Omega u)@ are pinned explicitly: the truth algebra no longer fixes the universe
  -- (the @tau -> u@ fundep is gone), so the universe is supplied at the quantifier.
  bigWedge @a @u @(Omega u) lp guard
    (binaryPredicate @u lp paramMLP)
