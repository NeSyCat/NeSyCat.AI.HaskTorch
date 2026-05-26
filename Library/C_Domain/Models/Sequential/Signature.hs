-- | The Sequential combinator — SIGNATURE: the ABSTRACT layer vocabulary (symbols
--   only, NO computation), the pure 'Arch' (a list of symbols), and composition.
--
--   Because the symbols carry no functions, an 'Arch' is fully analyzable DATA — you
--   can 'show' it, count it, walk it — and it admits MANY interpretations (run it as a
--   tensor net, count its parameters, shape-check it, draw it), exactly as a logical
--   signature's symbols admit Boolean / Tensor interpretations. The concrete meaning
--   of each symbol (what @ELU@ computes, how @Linear@ samples) lives in an
--   interpretation, never here.
module C_Domain.Models.Sequential.Signature
  ( Layer (..),
    Arch,
    (>>>),
    linearL,
    conv2dL,
    eluL,
    reluL,
    sigmoidL,
    maxPoolL,
    flattenL,
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

-- | Compose architectures: append (pure).
(>>>) :: Arch -> Arch -> Arch
(>>>) = (++)

infixr 5 >>>

-- pure symbol-builders (just the symbols — no functions) -------------------------

linearL :: Int -> Int -> Arch
linearL i o = [Linear i o]

conv2dL :: Int -> Int -> Int -> Arch
conv2dL i o k = [Conv2d i o k]

eluL, reluL, sigmoidL, maxPoolL, flattenL :: Arch
eluL = [ELU]
reluL = [ReLU]
sigmoidL = [Sigmoid]
maxPoolL = [MaxPool]
flattenL = [Flatten]
