{-# LANGUAGE TypeFamilies #-}

-- | The MODELS' SIGNATURE — abstract, a class only (no concrete weights, no bodies),
--   mirroring how "C_Domain.Signature" and "B_Logical.LogicalSignature" are pure
--   classes. A model is a parameter space (the horizontal sort) whose points act as
--   a map on tensors (the vertical sort). Each concrete network is an INSTANCE — its
--   interpretation, under "C_Domain.Models.Interpretations" — providing 'forward'
--   (the map) and 'fresh' (how to draw a fresh point of the space).
module C_Domain.Models.Signature (Model (..)) where

import Torch (Tensor)

class Model space where
  -- | What 'fresh' needs to build the weights (e.g. layer dimensions). A host type
  --   assigned per model — @(Int,Int,Int)@ for the MLP, @()@ for the fixed CNN —
  --   just like the domain assigns @type Point MeasU = (Float,Float)@.
  type Init space

  -- | @forward θ x@: run the network at parameter point θ on input x (logits out).
  forward :: space -> Tensor -> Tensor

  -- | @fresh cfg@: draw fresh random weights for the given init config.
  fresh :: Init space -> IO space
