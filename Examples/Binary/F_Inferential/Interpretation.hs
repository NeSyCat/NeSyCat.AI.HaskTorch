-- | Inference layer (F) — INTERPRETATION for the Binary example: ONLY the training
--   hyper-parameters. Binary's satisfaction is a @LogTens Bool@, so it REUSES the library's shared
--   @instance InferenceSignature (LogTens Bool)@ (@lossKnow = negLogSat@, in
--   "F_Inferential.InferenceInterpretation"), so there is no instance to declare here.
--   The generic objective @lossKnow . sat@ (in "Example") penalizes the @sat@ from D.
module Binary.F_Inferential.Interpretation (trainConfig) where

-- | (epochs, learning rate). The probabilistic (BCE) objective converges more slowly than
--   the old smoothed-fuzzy SAT, so use a larger step than the historical 0.001.
trainConfig :: (Int, Float)
trainConfig = (1000, 0.01)
