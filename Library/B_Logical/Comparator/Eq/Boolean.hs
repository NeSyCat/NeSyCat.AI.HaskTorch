-- | Equality predicate lifted to Boolean truth: (.==) via Haskell Eq.
module B_Logical.Comparator.Eq.Boolean ((.==)) where

infix 4 .==

-- | Point equality: a .== b = (a == b).
(.==) :: (Eq a) => a -> a -> Bool
x .== y = x == y
