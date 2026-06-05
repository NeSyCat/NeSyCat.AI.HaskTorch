{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | SIGNATURE A2Mon-BLat -- the "aggregated 2-monoid bounded lattice": 'TwoMonBLat'
--   (all the connector + comparator symbols, via the superclass) PLUS the
--   AGGREGATIONS (the quantifier / Quantor symbols). A logic used as a whole goes
--   through this class; connective-only code uses 'TwoMonBLat' directly.
--
--   The aggregation IS the Kleisli bind of the monad @m@ (the law of total probability for
--   @Dist@, the convolution for @LogVec@), so the class is parametrized over @m@ directly. The
--   point type @a@ sits at the class level too (so an interpretation may fix it -- e.g. the
--   @LogVec@ instance only quantifies over batch tensors, @a = Torch.Tensor@).
module B_Logical.Signature.A2MonBLat (A2MonBLat (..)) where

import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))

class (TwoMonBLat tau, Monad m) => A2MonBLat a m tau where
  -- == Quantor symbols ==

  -- lattice aggregations
  bigVee :: ParamsLogic tau -> Guard m a -> (a -> m tau) -> m tau -- ^ \bigvee  (\exists)
  bigWedge :: ParamsLogic tau -> Guard m a -> (a -> m tau) -> m tau -- ^ \bigwedge  (\forall)

  -- monoid aggregations
  bigOplus :: Guard m a -> (a -> m tau) -> m tau -- ^ \bigoplus
  bigOtimes :: Guard m a -> (a -> m tau) -> m tau -- ^ \bigotimes
