-- | Universal quantifier for Boolean truth (MeasU universe).
--
-- bigWedge = commutator (mapM phi guard) then foldl wedge True.
-- Semantics: forall x in guard. phi(x) = True iff phi holds for every element.
-- This is the classical "for all" over a finite list.
module B_Logical.Quantor.BigWedge.Boolean (bigWedge) where

import A_Categorical.Category.Monads.Dist ()  -- Monad instance for Dist

-- | Boolean universal quantifier over a finite list guard.
bigWedge :: () -> [a] -> (a -> IO Bool) -> IO Bool
bigWedge _ guard phi = do
    omegas <- mapM phi guard
    return (foldl (&&) True omegas)
