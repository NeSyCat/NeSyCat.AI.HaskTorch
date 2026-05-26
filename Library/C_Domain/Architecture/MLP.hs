-- | The Binary MLP ARCHITECTURE, plus its forward. 'mlpArch' is the pure 'Arch'
--   (the single source of truth); 'mlp' is its
--   forward at θ — @mlp = runArch mlpArch@, so a call site writes @mlp θ x@ instead of
--   @runArch mlpArch θ x@. Draw θ with @sampleWeights mlpArch@.
module C_Domain.Architecture.MLP (mlpArch, mlp) where

import C_Domain.Interpretation (Weights, runArch)
import C_Domain.Signature (Arch, Layer (..))
import Torch (Tensor)

-- | The Binary MLP: 2 -> 16 -> 16 -> 1, ELU between the linear layers.
mlpArch :: Arch
mlpArch = [Linear 2 16, ELU, Linear 16 16, ELU, Linear 16 1]

-- | The MLP forward at θ: @mlp θ x = runArch mlpArch θ x@.
mlp :: Weights -> Tensor -> Tensor
mlp = runArch mlpArch
