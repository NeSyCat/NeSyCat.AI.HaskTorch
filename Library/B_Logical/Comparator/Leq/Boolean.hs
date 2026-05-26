-- | Less-than-or-equal predicate lifted to Boolean truth: (.<=) via Haskell Ord.
module B_Logical.Comparator.Leq.Boolean ((.<=)) where

infix 4 .<=

-- | Less-than-or-equal: a .<= b = (a <= b).
(.<=) :: (Ord a) => a -> a -> Bool
x .<= y = x <= y
