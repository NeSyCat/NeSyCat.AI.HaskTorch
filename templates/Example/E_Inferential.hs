-- | E (inference) layer for the Template example: the knowledge-loss penalty.
--   Here it REUSES 'softplus' from the library; replace with your own (e.g. a
--   categorical NLL like the MNIST example) as needed.
module Examples.Template.E_Inferential (templateLoss) where

import Lib.E_Inferential.Library.Softplus (softplus)
import qualified Torch

-- | Map the grammatical axiom's value to a scalar loss.
templateLoss :: Torch.Tensor -> Torch.Tensor
templateLoss sat = Torch.mean (softplus sat)
