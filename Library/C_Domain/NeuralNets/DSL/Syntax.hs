-- | The architecture DSL — SIGNATURE: the ABSTRACT layer vocabulary (symbols
--   only, NO computation), the pure 'Arch' (a list of symbols composed in sequence),
--   and composition.
--
--   Because the symbols carry no functions, an 'Arch' is fully analyzable DATA — you
--   can 'show' it, count it, walk it — and it admits MANY interpretations (run it as a
--   tensor net, sample its weights, shape-check it, draw it), exactly as a logical
--   signature's symbols admit Boolean / Tensor interpretations. The concrete meaning
--   of each symbol (what @ELU@ computes, how @Linear@ samples) lives in an
--   interpretation, never here.
--
--   An architecture is written as a plain list literal, e.g.
--   @[Linear 2 16, ELU, Linear 16 1]@; use '(>>>)' to glue reusable sub-architectures.
module C_Domain.NeuralNets.DSL.Syntax
  ( Layer (..),
    Arch,
    (>>>),
  )
where

-- | The layer VOCABULARY — pure symbols (names + shapes), no computation. Add a kind
--   here, then give it meaning in each interpretation.
data Layer
  = Linear Int Int -- in-features, out-features
  | Conv2d Int Int Int -- in-channels, out-channels, square kernel
  | ELU
  | ReLU
  | Sigmoid
  | MaxPool
  | Flatten
  deriving (Show, Eq)

-- | An ARCHITECTURE is just a list of layer symbols — pure, analyzable data.
type Arch = [Layer]

-- | Glue architectures end to end (append). Handy for composing reusable blocks;
--   a single architecture is just a list literal.
(>>>) :: Arch -> Arch -> Arch
(>>>) = (++)

infixr 5 >>>
