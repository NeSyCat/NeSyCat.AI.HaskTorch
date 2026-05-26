-- | Tensor negation: neg = arithmetic negation (-x) in logit space.
--
-- In the logit encoding, negation is the additive inverse.
-- This is De Morgan compatible: neg (vee beta a b) = wedge beta (neg a) (neg b).
module B_Logical.Connector.Neg.Tensor (neg) where

import qualified Torch

-- | Additive inverse.  Involution: neg (neg x) = x.
neg :: Torch.Tensor -> Torch.Tensor
neg = negate
