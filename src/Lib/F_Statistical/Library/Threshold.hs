-- | threshold p t = p > t  (predicted probability exceeds the threshold).
module Lib.F_Statistical.Library.Threshold (threshold) where

threshold :: Double -> Double -> Bool
threshold p t = p > t
