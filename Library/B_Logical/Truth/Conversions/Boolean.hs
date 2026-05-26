-- | Conversion functions into Boolean truth (Omega = Bool).
module B_Logical.Truth.Conversions.Boolean (b2o, int2o) where

-- | Identity: Bool is already Omega.
b2o :: Bool -> Bool
b2o = id

-- | Integer to truth: 0 -> False, anything else -> True.
int2o :: Int -> Bool
int2o 0 = False
int2o _ = True
