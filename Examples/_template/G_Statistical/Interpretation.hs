-- | Statistics layer (G) — INTERPRETATION for the Template example: build a labeled
--   'Report' of whatever metrics make sense (any labels you like; printing/averaging
--   are generic). This stub reports a single placeholder metric.
module Template.G_Statistical.Interpretation (report, templateReport) where

import Template.C_Domain.Interpretation (Params)
import Template.E_Data.Signature (TemplateDataset)
import G_Statistical.Report (Report (..))

templateReport :: Params -> TemplateDataset -> Report
templateReport _theta _ds = Report [("TODO-metric", 0.0)]

-- | G-layer manifest piece for the Example.
report :: Params -> TemplateDataset -> IO Report
report theta ds = return (templateReport theta ds)
