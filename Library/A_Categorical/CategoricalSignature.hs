{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

module A_Categorical.CategoricalSignature
  ( Framework (..),
  )
where

import Data.Kind (Type)

-- | Semantic Framework: a label naming the Kleisli monad @M u@ that drives the
-- interpretation pipeline. The label (e.g. GeomU, MeasU) is the anchor every
-- interpretation class hangs off -- the truth object @Omega u@, @Guard u a@, the
-- logical connectives, the domain sorts/params are all separate classes keyed on
-- the SAME label, so selecting a framework selects that whole constellation.
--
-- M u : Type -> Type  (the Kleisli monad)
class Framework u where
  type M u :: Type -> Type
