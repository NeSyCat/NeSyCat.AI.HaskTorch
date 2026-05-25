-- | meanWhere vals mask = mean of the values where mask is True  (0 if none).
module Lib.F_Statistical.Library.MeanWhere (meanWhere) where

meanWhere :: [Double] -> [Bool] -> Double
meanWhere vals mask =
  let selected = [v | (v, True) <- zip vals mask]
   in if null selected
        then 0.0
        else sum selected / fromIntegral (length selected)
