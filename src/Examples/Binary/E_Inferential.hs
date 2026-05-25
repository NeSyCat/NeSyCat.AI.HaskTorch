-- | Inference layer for the Binary example: REUSE the library's TensReal
--   interpretation (so @lossKnow = softplus@). Re-exports the inference signature
--   and pulls in its interpretation. Swap here for a different penalty.
module Examples.Binary.E_Inferential
  ( module Lib.E_Inferential.InferenceSignature,
  )
where

import Lib.E_Inferential.InferenceInterpretation ()
import Lib.E_Inferential.InferenceSignature
