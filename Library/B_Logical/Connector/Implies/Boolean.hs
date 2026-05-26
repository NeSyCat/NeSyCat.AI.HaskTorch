-- | Classical Boolean implication: implies a b = not a || b
module B_Logical.Connector.Implies.Boolean (implies) where

-- | Material conditional.  Equivalent to @neg a \`vee\` b@.
implies :: () -> Bool -> Bool -> Bool
implies _ a b = not a || b
