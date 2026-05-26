-- | Classical Boolean additive monoid operation: oplus = (||)
--
-- In the Boolean case the additive monoid coincides with the lattice join (vee).
-- The neutral element is v0 = False.
module B_Logical.Connector.Oplus.Boolean (oplus) where

-- | Additive monoid operation on Bool.  oplus = (||).
oplus :: Bool -> Bool -> Bool
oplus = (||)
