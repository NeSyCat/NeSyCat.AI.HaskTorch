-- | Inference layer (F) — INTERPRETATION for the MNIST multi-digit example: ONLY the training
--   hyper-parameters. The satisfaction is a @LogTens Bool@, so it REUSES the library's shared
--   @instance InferenceSignature (LogTens Bool)@ (in "F_Inferential.InferenceInterpretation") --
--   there is no instance to declare here. The generic objective @lossKnow . sat@ (in "Example")
--   penalizes the @sat@ exported by D.
module MnistMultiDigit.F_Inferential.Interpretation (trainConfig) where

-- | (epochs, learning rate). Multi-digit is harder (all four digits must be right), like LTN's
--   multi-digit setup: 30 epochs, Adam 0.001.
trainConfig :: (Int, Float)
trainConfig = (30, 0.001)
