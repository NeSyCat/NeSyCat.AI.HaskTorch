{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

-- | Grammatical interpretation of BinaryFormulas in GeomU (TENS + Identity). Exports the
--   satisfaction 'binarySat' (the conjunction of the example's closed axioms) consumed by
--   the inference layer; the satisfaction lives in the logit truth object (@Torch.Tensor@).
module Binary.D_Grammatical.InterpretationTens
  ( binarySat,
    binaryAxiomTens,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.Tensor () -- TwoMonBLat GeomU Torch.Tensor (for satAgg)
import B_Logical.Library.SatAgg (satAgg)
import Binary.C_Domain.Signature (BinarySorts (..))
import Binary.C_Domain.Interpretation ()
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import Binary.D_Grammatical.Signature (binarySentence)
import Data.Functor.Identity (runIdentity)
import qualified Torch

-- | Binary axiom in GeomU (TENS + Identity). Guard is a batch tensor (a finite subset).
binaryAxiomTens :: Torch.Tensor -> Torch.Tensor -> Weights -> Omega GeomU
binaryAxiomTens betaT guard paramMLP =
  runIdentity (binarySentence @GeomU betaT guard paramMLP)

-- | The knowledge-base satisfaction exported to inference: the conjunction ('satAgg') of
--   Binary's closed axioms (here one), at smoothing @beta = 1.75@. @binarySat theta data@.
binarySat :: Weights -> Torch.Tensor -> Torch.Tensor
binarySat theta dataT = satAgg [binaryAxiomTens (Torch.asTensor (1.75 :: Float)) dataT theta]
