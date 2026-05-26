-- | Tensor bottom: bot = -infinity (logit space).
--
-- In the logit encoding, -inf represents certainty of falsehood.
-- It is the neutral element of smooth max (LogSumExp): vee(-inf, x) = x.
module B_Logical.Connector.Bot.Tensor (bot) where

import qualified Torch

-- | Least element: -∞ as a rank-1 tensor of shape [1].
bot :: Torch.Tensor
bot = Torch.asTensor [(-1.0 / 0.0) :: Float]
