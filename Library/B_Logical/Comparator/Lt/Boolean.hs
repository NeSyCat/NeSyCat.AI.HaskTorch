-- | Strict less-than predicate lifted to Boolean truth: (.<) via Haskell Ord.
module B_Logical.Comparator.Lt.Boolean ((.<)) where

infix 4 .<

-- | Strict less-than: a .< b = (a < b).
(.<) :: (Ord a) => a -> a -> Bool
x .< y = x < y
