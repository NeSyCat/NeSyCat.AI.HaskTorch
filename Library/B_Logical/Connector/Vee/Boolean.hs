-- | Classical Boolean disjunction: vee = (||)
module B_Logical.Connector.Vee.Boolean (vee) where

-- | Logical OR.  ParamsLogic Bool = () so the parameter is discarded.
vee :: () -> Bool -> Bool -> Bool
vee _ = (||)
