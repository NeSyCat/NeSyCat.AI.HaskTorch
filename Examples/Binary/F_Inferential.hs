-- | Inference layer (F) for the Binary example: the training config and the
--   objective — the inference penalty (@lossKnow@ = softplus, reused from the
--   library) of the grammatical (D) axiom evaluated over the (E) data. Swap the
--   penalty here for a different inference interpretation.
module Binary.F_Inferential (objective, trainConfig) where

import Binary.D_Grammatical.InterpretationTens (binaryAxiomTens)
import Binary.E_Data.Loader (BinaryDataset (..))
import C_Domain.Models.MLP (ParamsMLP)
import F_Inferential.InferenceInterpretation ()
import F_Inferential.InferenceSignature (InferenceSignature (..))
import qualified Torch

-- | (epochs, learning rate).
trainConfig :: (Int, Float)
trainConfig = (1000, 0.001)

-- | The loss the trainer minimizes: lossKnow (softplus) of the GeomU axiom over
--   the training points (beta = 1.75). Reaches the net only through the axiom.
objective :: BinaryDataset -> ParamsMLP -> Torch.Tensor
objective ds theta = lossKnow (binaryAxiomTens (Torch.asTensor (1.75 :: Float)) (trainData ds) theta)
