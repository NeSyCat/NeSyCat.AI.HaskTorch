-- | C (domain) interpretation for the Template example: assign the signature's
--   sorts/symbols to concrete objects/morphisms per universe (MeasU, GeomU), and
--   expose the parameter space (horizontal sort) as the canonical 'Params' +
--   'initParams'. See "Binary.C_Domain.Interpretation" for a worked example.
--
--   The MODEL is just an 'Arch' (a sequence of layers); θ is its PURE 'Weights'.
--   This stub picks a trivial 1->1 architecture so the scaffold builds and runs —
--   reuse a shared one from "C_Domain.Architecture.*" (e.g. @mlpArch@) or
--   build your own with @linearL@/@conv2dL@/@reluL@/@(>>>)@.
module Template.C_Domain.Interpretation
  ( Params,
    initParams,
    forwardTemplate,
  )
where

import C_Domain.Interpretation (Weights, runArch, sampleWeights)
import C_Domain.Signature (Arch, linearL)
import qualified Torch

-- ============================================================
--  Parameter spaces (horizontal sorts): Theta = the pure weights of 'arch'
-- ============================================================

-- | This example's architecture — choose it here (trivial 1->1 stub).
arch :: Arch
arch = linearL 1 1

-- | The (chosen) horizontal sort: the PURE parameters (weights) of 'arch'.
type Params = Weights

-- | Draw the initial theta_0 — fresh weights for 'arch'.
initParams :: IO Params
initParams = sampleWeights arch

-- | Run the chosen architecture at θ. (In a real example this is reached only
--   through a domain symbol like @classifierA@/@digit@; the stub uses it directly.)
forwardTemplate :: Weights -> Torch.Tensor -> Torch.Tensor
forwardTemplate = runArch arch
