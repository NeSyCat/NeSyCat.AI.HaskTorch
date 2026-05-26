-- | Inequality predicate lifted to Boolean truth: (./=) via Haskell Eq.
module B_Logical.Comparator.Neq.Boolean ((./=)) where

infix 4 ./=

-- | Point inequality: a ./= b = (a /= b).
(./=) :: (Eq a) => a -> a -> Bool
x ./= y = x /= y
