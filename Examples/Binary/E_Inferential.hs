-- | Inference layer for the Binary example: REUSE the library's TensReal
--   interpretation (so @lossKnow = softplus@). Re-exports the inference signature
--   and pulls in its interpretation. Swap here for a different penalty.
module Binary.E_Inferential
  ( module E_Inferential.InferenceSignature,
  )
where

import E_Inferential.InferenceInterpretation ()
import E_Inferential.InferenceSignature
