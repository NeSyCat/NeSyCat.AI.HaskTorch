-- | Logic layer for the Binary example: REUSE Boolean (MeasU) + Tensor/TensReal
--   (GeomU). Re-exports the logical signatures and pulls in their interpretations
--   (so the instances are in scope). Swap here for a different logic.
module Binary.B_Logical
  ( module B_Logical.LogicalSignature,
    module B_Logical.LogicalQuantSignature,
  )
where

import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.Tensor ()
import B_Logical.LogicalQuantSignature
import B_Logical.LogicalSignature
