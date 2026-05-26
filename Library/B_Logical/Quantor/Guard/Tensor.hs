{-# LANGUAGE TypeFamilies #-}

-- | Guard type for the GeomU universe: batch tensors.
--
-- A guard for smooth tensor quantification is a batch tensor of shape [N, ...].
-- The predicate phi :: Torch.Tensor -> Identity Omega is applied once and
-- PyTorch broadcasting distributes it over the N elements.
module B_Logical.Quantor.Guard.Tensor where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.LogicalQuantSignature (Guard)
import qualified Torch

-- | Guard GeomU Torch.Tensor = Torch.Tensor  (batch tensor of shape [N, ...])
type instance Guard GeomU Torch.Tensor = Torch.Tensor
