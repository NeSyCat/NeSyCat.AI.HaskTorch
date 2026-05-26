-- | Multiplicative quantifier for Boolean truth (MeasU universe).
--
-- In the Boolean interpretation, bigOtimes collapses to bigWedge:
-- the multiplicative monoid on Bool is (&&, True), identical to the lattice meet.
-- bigOtimes guard phi = forall x in guard. phi(x)
module B_Logical.Quantor.BigOtimes.Boolean (bigOtimes) where

import A_Categorical.Category.Monads.Dist ()  -- Monad instance for Dist

-- | Boolean multiplicative quantifier (= universal) over a finite list guard.
bigOtimes :: [a] -> (a -> IO Bool) -> IO Bool
bigOtimes guard phi = do
    omegas <- mapM phi guard
    return (foldl (&&) True omegas)
