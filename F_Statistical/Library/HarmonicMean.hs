-- | harmonicMean a b = 2ab/(a+b)  (0 if a+b <= 0).
module F_Statistical.Library.HarmonicMean (harmonicMean) where

harmonicMean :: Double -> Double -> Double
harmonicMean a b
  | a + b > 0 = 2.0 * a * b / (a + b)
  | otherwise = 0.0
