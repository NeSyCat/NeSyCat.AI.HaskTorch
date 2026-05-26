-- | Existential quantifier for Boolean truth (MeasU universe).
--
-- bigVee = commutator (mapM phi guard) then foldl vee False.
-- Semantics: exists x in guard. phi(x) = True iff phi holds for at least one element.
-- This is the classical "there exists" over a finite list.
module B_Logical.Quantor.BigVee.Boolean (bigVee) where

import A_Categorical.Category.Monads.Dist ()  -- Monad instance for Dist

-- | Boolean existential quantifier over a finite list guard.
bigVee :: () -> [a] -> (a -> IO Bool) -> IO Bool
bigVee _ guard phi = do
    omegas <- mapM phi guard
    return (foldl (||) False omegas)
