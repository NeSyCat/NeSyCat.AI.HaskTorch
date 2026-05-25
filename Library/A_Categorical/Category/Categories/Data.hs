{-# LANGUAGE FlexibleInstances #-}

-- | Objects of the DATA category (set/measure paradigm).
module A_Categorical.Category.Categories.Data
  ( DataObj,
  )
where

import Numeric.Natural (Natural)

-- | Type membership in the DATA type system.
class DataObj a

instance (DataObj a, DataObj b) => DataObj (a, b)

instance (DataObj a) => DataObj [a]

instance DataObj Bool

instance DataObj Natural

instance DataObj Integer

instance DataObj Char

instance DataObj Double

instance DataObj Float

instance DataObj Int

instance DataObj ()
