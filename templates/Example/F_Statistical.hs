-- | F (statistics) layer for the Template example: build a labeled 'Report' of
--   whatever metrics make sense (any labels you like; printing/averaging are
--   generic). This stub reports a single placeholder metric.
module Examples.Template.F_Statistical (templateReport) where

import Examples.Template.C_Domain.Model (ParamsTemplate)
import Examples.Template.G_Data.Loader (TemplateDataset)
import Lib.F_Statistical.Report (Report (..))

templateReport :: ParamsTemplate -> TemplateDataset -> Report
templateReport _theta _ds = Report [("TODO-metric", 0.0)]
