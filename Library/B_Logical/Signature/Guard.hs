{-# LANGUAGE TypeFamilies #-}

-- | The subset a guarded quantifier ranges over, per universe @u@ and point type @a@
--   (e.g. @Guard MeasU a = [a]@). Shared by every signature that has aggregations.
module B_Logical.Signature.Guard (Guard) where

import Data.Kind (Type)

type family Guard u a :: Type
