-- | threshold p t = p > t  (predicted probability exceeds the threshold).
module F_Statistical.Library.Threshold (threshold) where

threshold :: Double -> Double -> Bool
threshold p t = p > t
