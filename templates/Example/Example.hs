{-# LANGUAGE TypeFamilies #-}

-- | The Template example: the full A–G stack assembled into one 'Example'
--   instance. This stub builds and runs as-is (@nesycat <name> 1@); fill in the
--   C/D/E/F/G layers in this folder to make it a real experiment.
module Examples.Template.Example (Template) where

import Examples.Template.A_Categorical ()
import Examples.Template.B_Logical ()
import Examples.Template.C_Domain.Model (ParamsTemplate, ParamsTemplateSpec (..), forwardTemplate)
import Examples.Template.E_Inferential (templateLoss)
import Examples.Template.F_Statistical (templateReport)
import Examples.Template.G_Data.Loader (TemplateDataset (..), loadTemplateDataset)
import Lib.Example (Example (..))

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
