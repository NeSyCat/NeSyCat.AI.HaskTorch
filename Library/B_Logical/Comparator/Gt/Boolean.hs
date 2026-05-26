-- | Strict greater-than predicate lifted to Boolean truth: (.>) via Haskell Ord.
module B_Logical.Comparator.Gt.Boolean ((.>)) where

infix 4 .>

-- | Strict greater-than: a .> b = (a > b).
(.>) :: (Ord a) => a -> a -> Bool
x .> y = x > y
