-- | Classical Boolean negation: neg = not
module B_Logical.Connector.Neg.Boolean (neg) where

-- | Logical NOT.  Involution: neg (neg x) = x.
neg :: Bool -> Bool
neg = not
