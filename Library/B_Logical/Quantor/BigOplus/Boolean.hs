-- | Additive quantifier for Boolean truth (MeasU universe).
--
-- In the Boolean interpretation, bigOplus collapses to bigVee:
-- the additive monoid on Bool is (||, False), identical to the lattice join.
-- bigOplus guard phi = exists x in guard. phi(x)
module B_Logical.Quantor.BigOplus.Boolean (bigOplus) where

import A_Categorical.Category.Monads.Dist ()  -- Monad instance for Dist

-- | Boolean additive quantifier (= existential) over a finite list guard.
bigOplus :: [a] -> (a -> IO Bool) -> IO Bool
bigOplus guard phi = do
    omegas <- mapM phi guard
    return (foldl (||) False omegas)
