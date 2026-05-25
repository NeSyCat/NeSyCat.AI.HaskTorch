-- | F (statistics) layer for the Template example: build a labeled 'Report' of
--   whatever metrics make sense (any labels you like; printing/averaging are
--   generic). This stub reports a single placeholder metric.
module Template.G_Statistical (report, templateReport) where

import Template.C_Domain.Model (ParamsTemplate)
import Template.E_Data.Loader (TemplateDataset)
import G_Statistical.Report (Report (..))

templateReport :: ParamsTemplate -> TemplateDataset -> Report
templateReport _theta _ds = Report [("TODO-metric", 0.0)]

-- | G-layer manifest piece for the Example.
report :: ParamsTemplate -> TemplateDataset -> IO Report
report theta ds = return (templateReport theta ds)
