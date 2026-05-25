{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

-- | C (domain) model for the Template example: the neural component. This stub
--   is a trivial 1->1 linear map so the scaffold builds and runs; replace it with
--   your architecture (or reuse one from "Lib.Models").
module Examples.Template.C_Domain.Model
  ( ParamsTemplate (..),
    ParamsTemplateSpec (..),
    forwardTemplate,
  )
where

import GHC.Generics (Generic)
import Torch (Linear, LinearSpec (..), Parameterized, Randomizable (..), Tensor, linear)

data ParamsTemplateSpec = ParamsTemplateSpec deriving (Show, Eq)

newtype ParamsTemplate = ParamsTemplate {lin :: Linear}
  deriving (Generic, Show, Parameterized)

instance Randomizable ParamsTemplateSpec ParamsTemplate where
  sample ParamsTemplateSpec = ParamsTemplate <$> sample (LinearSpec 1 1)

forwardTemplate :: ParamsTemplate -> Tensor -> Tensor
forwardTemplate ParamsTemplate {..} = linear lin
