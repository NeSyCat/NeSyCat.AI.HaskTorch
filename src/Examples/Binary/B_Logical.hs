-- | Logic layer for the Binary example: REUSE Boolean (MeasU) + Tensor/TensReal
--   (GeomU). Re-exports the logical signatures and pulls in their interpretations
--   (so the instances are in scope). Swap here for a different logic.
module Examples.Binary.B_Logical
  ( module Lib.B_Logical.LogicalSignature,
    module Lib.B_Logical.LogicalQuantSignature,
  )
where

import Lib.B_Logical.Interpretations.Boolean ()
import Lib.B_Logical.Interpretations.Tensor ()
import Lib.B_Logical.LogicalQuantSignature
import Lib.B_Logical.LogicalSignature
