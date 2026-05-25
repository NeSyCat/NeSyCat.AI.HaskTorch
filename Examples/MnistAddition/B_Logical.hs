-- | Logic layer for the MNIST example: REUSE Boolean (MeasU) + Tensor/TensReal
--   (GeomU). Re-exports the logical signatures and pulls in their interpretations.
module MnistAddition.B_Logical
  ( module B_Logical.LogicalSignature,
    module B_Logical.LogicalQuantSignature,
  )
where

import B_Logical.Interpretations.Boolean ()
import B_Logical.Interpretations.Tensor ()
import B_Logical.LogicalQuantSignature
import B_Logical.LogicalSignature
