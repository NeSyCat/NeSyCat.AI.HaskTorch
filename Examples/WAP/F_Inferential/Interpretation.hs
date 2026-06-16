-- | Inference layer (F) -- INTERPRETATION for WAP: ONLY the training hyper-parameters,
--   following the reference protocol (DeepProbLog's @wap.py@: Adam, lr = 0.005, batch 10,
--   40 epochs; DeepStochLog re-ran all three systems under the same regime). WAP's
--   satisfaction is a @LogTens Bool@, so it REUSES the library's shared
--   @instance InferenceSignature (LogTens Bool)@ (@lossKnow = negLogSat@ -- exactly the
--   reference's @-log P(query)@ cross-entropy).
module WAP.F_Inferential.Interpretation (trainConfig) where

-- | (epochs, learning rate). 12 epochs for quick iteration (the loss converges by ~epoch
--   6); the citable reference protocol is 40 (DeepProbLog's @wap.py@: Adam, lr = 0.005,
--   batch 10, 40 epochs).
trainConfig :: (Int, Float)
trainConfig = (12, 0.005)
