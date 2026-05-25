-- | threshold p t = p > t  (predicted probability exceeds the threshold).
module G_Statistical.Library.Threshold (threshold) where

threshold :: Double -> Double -> Bool
threshold p t = p > t
