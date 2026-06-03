{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | SIGNATURE A2Mon-BLat -- the "aggregated 2-monoid bounded lattice": 'TwoMonBLat'
--   (all the connector + comparator symbols, via the superclass) PLUS the
--   AGGREGATIONS (the quantifier / Quantor symbols). A logic used as a whole goes
--   through this class; connective-only code uses 'TwoMonBLat' directly.
--
--   The point type @a@ sits at the class level (so an interpretation may fix it --
--   e.g. the GeomU instance only quantifies over batch tensors, @a = Torch.Tensor@).
module B_Logical.Signature.A2MonBLat (A2MonBLat (..)) where

import A_Categorical.CategoricalSignature (Framework (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))

class (TwoMonBLat tau, Framework u, Monad (M u)) => A2MonBLat a u tau where
  -- == Quantor symbols ==

  -- lattice aggregations
  bigVee :: ParamsLogic tau -> Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigvee  (\exists)
  bigWedge :: ParamsLogic tau -> Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigwedge  (\forall)

  -- monoid aggregations
  bigOplus :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigoplus
  bigOtimes :: Guard u a -> (a -> M u tau) -> M u tau -- ^ \bigotimes
