-- | Entailment order on Boolean truth values: vdash = (<=)
--
-- Compares two Omega values and returns a meta-truth (Bool).
-- False <= True, so False |= True (the lattice order on {False, True}).
module B_Logical.Comparator.Vdash.Boolean (vdash) where

-- | Boolean entailment: a |= b iff a <= b in {False < True}.
vdash :: Bool -> Bool -> Bool
vdash = (<=)
