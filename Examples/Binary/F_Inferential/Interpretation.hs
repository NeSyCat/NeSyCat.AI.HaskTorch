-- | Inference layer (F) — INTERPRETATION for the Binary example: the training
--   config and the objective. The inference penalty is 'lossKnow' interpreted as
--   'softplus' (REUSED from the library 'F_Inferential.InferenceInterpretation');
--   the objective applies it to the D (grammatical) axiom over the E data. Reaches
--   the net only through the axiom.
module Binary.F_Inferential.Interpretation (objective, trainConfig) where

import Binary.D_Grammatical.InterpretationTens (binaryAxiomTens)
import Binary.E_Data.Signature (BinaryDataset (..))
import C_Domain.Models.Sequential.Interpretation (Weights)
import F_Inferential.InferenceInterpretation ()
import F_Inferential.InferenceSignature (InferenceSignature (..))
import qualified Torch

-- | (epochs, learning rate).
trainConfig :: (Int, Float)
trainConfig = (1000, 0.001)

-- | The loss the trainer minimizes: lossKnow (softplus) of the GeomU axiom over
--   the training points (beta = 1.75).
objective :: BinaryDataset -> Weights -> Torch.Tensor
objective ds theta = lossKnow (binaryAxiomTens (Torch.asTensor (1.75 :: Float)) (trainData ds) theta)
