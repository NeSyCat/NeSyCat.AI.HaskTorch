-- | Inference layer (F) -- INTERPRETATION for WAP: ONLY the training hyper-parameters,
--   following the reference protocol (DeepProbLog's @wap.py@: Adam, lr = 0.005, batch 10,
--   40 epochs; DeepStochLog re-ran all three systems under the same regime). WAP's
--   satisfaction is a @LogVec Bool@, so it REUSES the library's shared
--   @instance InferenceSignature (LogVec Bool)@ (@lossKnow = negLogSat@ -- exactly the
--   reference's @-log P(query)@ cross-entropy).
module WAP.F_Inferential.Interpretation (trainConfig) where

-- | (epochs, learning rate) -- the reference protocol (DeepProbLog's @wap.py@: Adam,
--   lr = 0.005, batch 10, 40 epochs). The loss converges by ~epoch 6, so ~12 epochs
--   suffice for a quick check.
trainConfig :: (Int, Float)
trainConfig = (40, 0.005)
