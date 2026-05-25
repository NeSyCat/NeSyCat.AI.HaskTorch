-- | C (domain) interpretation for the Template example: assign the signature's
--   sorts/symbols to concrete objects/morphisms per universe (MeasU, GeomU), and
--   expose the parameter space (horizontal sort) as the canonical 'Params' +
--   'initParams' (using the model from "Template.C_Domain.Model"). See
--   "Binary.C_Domain.Interpretation" for a worked example.
module Template.C_Domain.Interpretation
  ( module Template.C_Domain.Model,
    Params,
    initParams,
  )
where

import Template.C_Domain.Model
import qualified Torch

-- ============================================================
--  Parameter spaces (horizontal sorts): Theta = this example's model weights
-- ============================================================

-- | The (chosen) horizontal sort: the parameter space this domain learns over.
type Params = ParamsTemplate

-- | Draw the initial theta_0 (hides HaskTorch's Randomizable / 'sample').
initParams :: IO Params
initParams = Torch.sample ParamsTemplateSpec
