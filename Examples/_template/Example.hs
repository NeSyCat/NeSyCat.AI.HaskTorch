{-# LANGUAGE TypeFamilies #-}

-- | The Template example: the full A–G stack assembled into one 'Example'
--   instance. This stub builds and runs as-is (@nesycat <name> 1@); fill in the
--   C/D/E/F/G layers in this folder to make it a real experiment.
module Template.Example (Template) where

import Template.A_Categorical ()
import Template.B_Logical ()
import Template.C_Domain.Model (ParamsTemplate, ParamsTemplateSpec (..), forwardTemplate)
import Template.E_Inferential (templateLoss)
import Template.F_Statistical (templateReport)
import Template.G_Data.Loader (TemplateDataset (..), loadTemplateDataset)
import Example (Example (..))

data Template

instance Example Template where
  type Params Template = ParamsTemplate
  type Spec Template = ParamsTemplateSpec
  type Data Template = TemplateDataset

  name = "Template (fill me in)"
  spec = ParamsTemplateSpec
  trainConfig = (10, 0.01)
  loadData = loadTemplateDataset
  objective ds theta = templateLoss (forwardTemplate theta (inputs ds))
  report theta ds = return (templateReport theta ds)
