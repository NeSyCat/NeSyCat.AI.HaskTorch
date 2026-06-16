{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

-- | Non-logical signature for Binary classification.
--
--   The sorts are monad-INVARIANT plain types: a @Point@ is a @Torch.Tensor@ in BOTH readings
--   (@Dist@ reasons on a single @[2]@ point, @LogTens@ on a @[B,2]@ batch), and the truth
--   object is @Bool@. So there is no per-monad sort assignment and no @encPoint@ bridge. The
--   only monad-dependent symbols are the Kleisli relations 'labelA' and 'classifierA' (their
--   monad @m@ is @Dist@ or @LogTens@).
module Binary.C_Domain.Signature
  ( Point,
    Omega,
    BinaryRel (..),
    BinaryKlRel (..),
  )
where

import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (universe-invariant plain types).
type Point = Torch.Tensor -- a point in R^2 (a [2] tensor, or a [B,2] batch)
type Omega = Bool         -- the truth object

-- | The ground-truth label relation -- CERTAIN but Kleisli (@m Omega@), so the truth flows
--   through the monad (@Dist@: one point at a time; @LogTens@: the whole batch in the leaf weights).
class (Monad m) => BinaryRel m where
  labelA :: Point -> m Omega

-- | The neural classifier as a parametrized Kleisli relation -- the genuinely per-monad
--   symbol (its monad @m@ is @Dist@ vs @LogTens@).
class (BinaryRel m) => BinaryKlRel m where
  classifierA :: Weights -> Point -> m Omega
