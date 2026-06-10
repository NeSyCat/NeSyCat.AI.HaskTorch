-- | Inference layer (F) — INTERPRETATION for the MNIST example: ONLY the training
--   hyper-parameters. MNIST's satisfaction is a @LogVec Bool@, so it REUSES the library's
--   shared @instance InferenceSignature (LogVec Bool)@ (@lossKnow = negLogSat@, in
--   "F_Inferential.InferenceInterpretation") -- there is no instance to declare here. The
--   generic objective @lossKnow . sat@ (in "Example") penalizes the @sat@ exported by D.
module MnistAddition.F_Inferential.Interpretation (trainConfig) where

-- | (epochs, learning rate). LTN's small-data single-digit setup: 20 epochs.
trainConfig :: (Int, Float)
trainConfig = (20, 0.001)
