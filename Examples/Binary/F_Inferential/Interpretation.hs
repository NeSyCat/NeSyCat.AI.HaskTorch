-- | Inference layer (F) — INTERPRETATION for the Binary example: ONLY the training
--   hyper-parameters. Binary REUSES the library's logit-truth loss instance
--   (@instance InferenceSignature Torch.Tensor@ with @lossKnow = softplus@, in
--   "F_Inferential.InferenceInterpretation"), so there is no instance to declare here.
--   The generic objective @lossKnow . sat@ (in "Example") penalizes the @sat@ from D.
module Binary.F_Inferential.Interpretation (trainConfig) where

-- | (epochs, learning rate).
trainConfig :: (Int, Float)
trainConfig = (1000, 0.001)
