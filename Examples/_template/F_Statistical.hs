-- | F (statistics) layer for the Template example: build a labeled 'Report' of
--   whatever metrics make sense (any labels you like; printing/averaging are
--   generic). This stub reports a single placeholder metric.
module Template.F_Statistical (templateReport) where

import Template.C_Domain.Model (ParamsTemplate)
import Template.G_Data.Loader (TemplateDataset)
import F_Statistical.Report (Report (..))

templateReport :: ParamsTemplate -> TemplateDataset -> Report
templateReport _theta _ds = Report [("TODO-metric", 0.0)]
