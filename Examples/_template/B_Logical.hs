-- | B (logic) layer for the Template example: REUSE Boolean (MeasU) +
--   Tensor/TensReal (GeomU). Swap here for a different logic.
module Template.B_Logical
  ( module B_Logical.LogicalSignature,
    module B_Logical.LogicalQuantSignature,
  )
where

import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.Tensor ()
import B_Logical.LogicalQuantSignature
import B_Logical.LogicalSignature
