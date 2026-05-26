-- | Classical Boolean multiplicative monoid operation: otimes = (&&)
--
-- In the Boolean case the multiplicative monoid coincides with the lattice meet (wedge).
-- The neutral element is v1 = True.
module B_Logical.Connector.Otimes.Boolean (otimes) where

-- | Multiplicative monoid operation on Bool.  otimes = (&&).
otimes :: Bool -> Bool -> Bool
otimes = (&&)
