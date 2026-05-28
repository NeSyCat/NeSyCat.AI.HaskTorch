{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | Logical interpretation: tensor-valued logic on \mathbb{R} (Omega = \mathbb{R}^1)
--   in the GeomU universe. All operations work on logits in \mathbb{R} (no sigmoid):
--   neg = -x, vee = smooth max (LogSumExp), wedge = its De Morgan dual,
--   top = +\infty, bot = -\infty. ParamsLogic = the beta smoothing tensor.
--   Instantiates 'TwoMonBLat' (connectives) and 'A2MonBLat' (quantifiers).
module B_Logical.Interpretations.Tensor
  ( module B_Logical.Interpretations.Tensor,
    module B_Logical.Signature.TwoMonBLat,
    module B_Logical.Signature.A2MonBLat,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import Data.Functor.Identity (Identity (..), runIdentity)
import qualified Torch
import qualified Torch.Functional.Internal as F

-- | Omega := I(tau) = \mathbb{R}^1 (a 1-element tensor)
type Omega = Torch.Tensor

-- | In GeomU the guard IS the batched data itself (the vectorized predicate is applied
--   to the whole batch, then reduced) -- polymorphic in the point type, mirroring
--   @Guard MeasU a = [a]@. So quantifying over single tensors (Binary) or tuples (MNIST)
--   needs no per-shape Guard instance.
type instance Guard GeomU a = a

------------------------------------------------------
-- TwoMonBLat: connectives on Omega (R-valued, logit space)
------------------------------------------------------

instance TwoMonBLat GeomU Omega where
  type ParamsLogic Omega = Torch.Tensor
  -- bounded lattice
  top = Torch.asTensor [(1.0 / 0.0) :: Float]
  bot = Torch.asTensor [(-1.0 / 0.0) :: Float]
  neg = negate
  vee betaT a b =
    let pa = a `Torch.mul` betaT
        pb = b `Torch.mul` betaT
     in F.logaddexp pa pb `Torch.div` betaT
  -- the two monoids (units + residual). wedge/implies use the TwoMonBLat defaults.
  oplus = Torch.add
  o0 = Torch.asTensor [0.0 :: Float]
  otimes = Torch.mul
  o1 = Torch.asTensor [1.0 :: Float]
  -- comparator
  vdash a b = Torch.asValue a <= (Torch.asValue b :: Float)

------------------------------------------------------
-- A2MonBLat: quantifiers via smooth sup/inf (LogSumExp)
------------------------------------------------------

instance A2MonBLat Torch.Tensor GeomU Omega where
  -- bigWedge = forall = smooth inf = De Morgan of LogSumExp
  bigWedge betaT guard phi =
    let result = runIdentity (phi guard)
        n = head (Torch.shape guard)
        negResult = neg result
        lse = F.logsumexp (negResult `Torch.mul` betaT) 0 False
        reduced = negate ((lse `Torch.sub` Torch.log (Torch.asTensor (fromIntegral n :: Float))) `Torch.div` betaT)
     in Identity (Torch.reshape [1] reduced)
  -- bigVee = exists = LogSumExp
  bigVee betaT guard phi =
    let result = runIdentity (phi guard)
        n = head (Torch.shape guard)
        lse = F.logsumexp (result `Torch.mul` betaT) 0 False
        reduced = (lse `Torch.sub` Torch.log (Torch.asTensor (fromIntegral n :: Float))) `Torch.div` betaT
     in Identity (Torch.reshape [1] reduced)
  bigOplus _ _ = error "bigOplus over GeomU not yet supported"
  bigOtimes _ _ = error "bigOtimes over GeomU not yet supported"

------------------------------------------------------
-- Internal Helpers
------------------------------------------------------

-- | Numerically stable log-sigmoid: log sigma(x) = -log(1 + exp(-x)).
logSigmoid :: Torch.Tensor -> Torch.Tensor
logSigmoid x = negate (Torch.log (Torch.exp (negate x) `Torch.add` Torch.onesLike x))

one :: Torch.Tensor
one = Torch.toDevice (Torch.Device Torch.CPU 0) $ Torch.asTensor (1.0 :: Float)

eps :: Torch.Tensor
eps = Torch.toDevice (Torch.Device Torch.CPU 0) $ Torch.asTensor (1.0e-8 :: Float)
