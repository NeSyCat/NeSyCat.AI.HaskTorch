-- | C (domain) interpretation for the Template example: assign the signature's
--   sorts/symbols to concrete objects/morphisms per universe (MeasU, GeomU), and
--   expose the model this domain uses as the canonical 'Params'/'Spec'/'spec'
--   (re-exported from "Template.C_Domain.Model"). See
--   "Binary.C_Domain.Interpretation" for a worked example.
module Template.C_Domain.Interpretation
  ( module Template.C_Domain.Model,
    Params,
    Spec,
    spec,
  )
where

import Template.C_Domain.Model

-- | C-layer manifest pieces for the Example: the model this domain uses.
type Params = ParamsTemplate

type Spec = ParamsTemplateSpec

spec :: Spec
spec = ParamsTemplateSpec
