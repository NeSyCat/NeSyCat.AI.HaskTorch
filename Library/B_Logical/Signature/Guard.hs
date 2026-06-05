{-# LANGUAGE TypeFamilies #-}

-- | The subset a guarded quantifier ranges over, per monad @m@ and point type @a@
--   (e.g. @Guard Dist a = [a]@, @Guard LogVec a = a@). Shared by every signature that has
--   aggregations. Keyed on the monad itself -- the guard genuinely differs by monad (a list to
--   fold for @Dist@, the batched tensor for @LogVec@), so the key moves from tag to monad.
module B_Logical.Signature.Guard (Guard) where

import Data.Kind (Type)

type family Guard (m :: Type -> Type) a :: Type
