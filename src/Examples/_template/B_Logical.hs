-- | B (logic) layer for the Template example: REUSE Boolean (MeasU) +
--   Tensor/TensReal (GeomU). Swap here for a different logic.
module Examples.Template.B_Logical
  ( module Lib.B_Logical.LogicalSignature,
    module Lib.B_Logical.LogicalQuantSignature,
  )
where

import Lib.B_Logical.Interpretations.Boolean ()
import Lib.B_Logical.Interpretations.Tensor ()
import Lib.B_Logical.LogicalQuantSignature
import Lib.B_Logical.LogicalSignature
