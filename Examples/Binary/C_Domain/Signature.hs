{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

-- | Non-logical signature for Binary classification.
--
--   The sorts are universe-INVARIANT plain types: a @Point@ is a @Torch.Tensor@ in BOTH
--   universes (MeasU reasons on a single @[2]@ point, GeomU on a @[B,2]@ batch), and the truth
--   object is @Bool@. So there is no per-universe sort assignment and no @encPoint@ bridge. The
--   only universe-dependent symbols are the Kleisli relations 'labelA' and 'classifierA' (their
--   monad @M u@ is @Dist@ in MeasU, @LogVec@ in GeomU).
module Binary.C_Domain.Signature
  ( Point,
    Omega,
    BinaryRel (..),
    BinaryKlRel (..),
  )
where

import A_Categorical.CategoricalSignature (Framework (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (universe-invariant plain types).
type Point = Torch.Tensor -- a point in R^2 (a [2] tensor, or a [B,2] batch)
type Omega = Bool         -- the truth object

-- | The ground-truth label relation -- CERTAIN but Kleisli (@M u Omega@), so the truth flows
--   through the monad (MeasU: one point at a time; GeomU: the whole batch in the leaf weights).
class (Framework u, Monad (M u)) => BinaryRel u where
  labelA :: Point -> M u Omega

-- | The neural classifier as a parametrized Kleisli relation -- the genuinely per-universe
--   symbol (its monad @M u@ is @Dist@ vs @LogVec@).
class (BinaryRel u) => BinaryKlRel u where
  classifierA :: Weights -> Point -> M u Omega
