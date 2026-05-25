-- | Logic layer for the MNIST example: REUSE Boolean (MeasU) + Tensor/TensReal
--   (GeomU). Re-exports the logical signatures and pulls in their interpretations.
module Examples.MnistAddition.B_Logical
  ( module Lib.B_Logical.LogicalSignature,
    module Lib.B_Logical.LogicalQuantSignature,
  )
where

import Lib.B_Logical.Interpretations.Boolean ()
import Lib.B_Logical.Interpretations.Tensor ()
import Lib.B_Logical.LogicalQuantSignature
import Lib.B_Logical.LogicalSignature
