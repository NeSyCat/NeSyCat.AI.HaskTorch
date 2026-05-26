-- | Classical Boolean conjunction: wedge = (&&)
module B_Logical.Connector.Wedge.Boolean (wedge) where

-- | Logical AND.  ParamsLogic Bool = () so the parameter is discarded.
wedge :: () -> Bool -> Bool -> Bool
wedge _ = (&&)
