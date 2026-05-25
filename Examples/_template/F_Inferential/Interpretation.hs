-- | Inference layer (F) — INTERPRETATION for the Template example: the training
--   config and the objective. This stub REUSES 'softplus' from the library and, for
--   simplicity, maps the model output directly through the penalty; in a real
--   example route the penalty through the D (grammatical) axiom over the data (see
--   "Binary.F_Inferential.Interpretation" / "MnistAddition.F_Inferential.Interpretation").
module Template.F_Inferential.Interpretation (objective, trainConfig, templateLoss) where

import Template.C_Domain.Model (ParamsTemplate, forwardTemplate)
import Template.E_Data.Signature (TemplateDataset (..))
import F_Inferential.Library.Softplus (softplus)
import qualified Torch

-- | (epochs, learning rate).
trainConfig :: (Int, Float)
trainConfig = (10, 0.01)

-- | The loss the trainer minimizes.
objective :: TemplateDataset -> ParamsTemplate -> Torch.Tensor
objective ds theta = templateLoss (forwardTemplate theta (inputs ds))

-- | Map the grammatical axiom's value to a scalar loss.
templateLoss :: Torch.Tensor -> Torch.Tensor
templateLoss sat = Torch.mean (softplus sat)
