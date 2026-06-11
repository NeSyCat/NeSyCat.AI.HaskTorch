-- | Parameterized (parameter space \mathbb{R}^{v \cdot d}): an embedding table -- the free
--   linear map out of a finite token sort (a lookup, i.e. one-hot followed by a linear map).
--   'sampleEmbed' draws fresh \theta from the standard normal (torch.nn.Embedding's init);
--   'embed' is the forward @\theta -> Tensor -> Tensor@ (Int64 indices @[..]@ -> vectors
--   @[.., d]@) via the fused ATen @embedding@ kernel -- the SAME op @torch.nn.Embedding@
--   dispatches to, so reference PyTorch architectures port verbatim.
module C_Domain.NeuralNets.DSL.Library.Parameterized.Embedding (EmbedW (..), sampleEmbed, embed) where

import Torch (Parameter, Parameterized (..), Tensor, makeIndependent, randnIO', toDependent)
import qualified Torch.Functional.Internal as FI

-- | The embedding table \theta : @[vocab, dim]@.
newtype EmbedW = EmbedW Parameter deriving (Show)

instance Parameterized EmbedW where
  flattenParameters (EmbedW w) = [w]
  _replaceParameters (EmbedW w) = EmbedW <$> _replaceParameters w

sampleEmbed :: Int -> Int -> IO EmbedW
sampleEmbed v d = EmbedW <$> (makeIndependent =<< randnIO' [v, d])

embed :: EmbedW -> Tensor -> Tensor
embed (EmbedW w) idx = FI.embedding (toDependent w) idx (-1) False False
