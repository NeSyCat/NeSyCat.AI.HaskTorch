-- | fractionTrue bs = (number of True) / (total)  (0 if empty).
module Lib.F_Statistical.Library.FractionTrue (fractionTrue) where

fractionTrue :: [Bool] -> Double
fractionTrue bs =
  let n = length bs
      k = length (filter id bs)
   in if n > 0 then fromIntegral k / fromIntegral n else 0.0
