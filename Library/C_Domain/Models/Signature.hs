-- | The MODELS' SIGNATURE — abstract, a class only (no concrete weights, no bodies),
--   mirroring how "C_Domain.Signature" and "B_Logical.LogicalSignature" are pure
--   classes. A model is a parameter space (the horizontal sort) whose points act as
--   a map on tensors (the vertical sort). Each concrete network is an INSTANCE — its
--   interpretation, under "C_Domain.Models.Interpretations".
module C_Domain.Models.Signature (Model (..)) where

import Torch (Tensor)

class Model space where
  -- | @forward θ x@: run the network at parameter point θ on input x (logits out).
  forward :: space -> Tensor -> Tensor
