{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

-- | Non-logical signature for the Word Algebra Problems example (Roy & Roth's Common Core
--   set, the canonical NeSy string benchmark of differentiable-Forth / DeepProbLog /
--   DeepStochLog).
--
--   The sorts are monad-INVARIANT plain types; the raw input is TEXT (a tokenized problem),
--   not a tensor -- the C interpretation owns the collation (tokens -> embedding -> BiGRU).
--   PURE SYNTAX: the sorts and the symbol classes, nothing else. Two classes:
--
--     Fun   ('WapFun')   -- the plain function symbol 'evalSketch' (monad-free; interpreted
--                           once in the base category by the Interpretation);
--     KlFun ('WapKlFun') -- the neural Kleisli symbols, per monad: the deterministic trunk
--                           'repS' (lifted by \eta -- a CERTAIN Kleisli step, so one bound
--                           r is shared by all heads) and the four classification heads.
module WAP.C_Domain.Signature
  ( Problem,
    Problems,
    Rep,
    Numbers,
    Answer,
    Perm,
    Op,
    Omega,
    WapFun (..),
    WapKlFun (..),
  )
where

import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch

-- Sorts (monad-invariant plain types).

-- | One tokenized problem: vocab ids + the positions of the three @\<NR\>@ number tokens.
type Problem = ([Int], [Int])

-- | The batched carrier of 'Problem' (a list -- the @Dist@ reading passes a singleton,
--   the @LogVec@ reading the whole mini-batch; the symbol's interpretation collates).
type Problems = [Problem]

-- | The trunk representation of a problem (monad-invariant, like @Image@ for MNIST: a
--   @[4096]@ vector per problem, @[B, 4096]@ for a batch).
type Rep = Torch.Tensor

-- | The three numbers of a problem, in text order (Prolog-exact INTEGER arithmetic).
type Numbers = (Int, Int, Int)

-- | The observed numeric answer (an integer; division is exact in the sketch grammar).
type Answer = Int

type Perm = Int -- 0..5: the six orderings of the three numbers

type Op = Int -- 0..3: plus, minus, times, div (exact, guarded)

type Omega = Bool -- the truth object

-- | Fun -- the plain function symbols of the domain (monad-free; declared here,
--   interpreted in "WAP.C_Domain.Interpretation"):
--
--     * 'evalSketch' -- the task's program space (an ordering of the three numbers, a first
--       operation, an optional swap with the third, a second operation); @Nothing@ = the
--       sketch FAILS on these numbers.
--     * 'repF' -- the COMPUTATIONAL function symbol: the deterministic neural trunk
--       (collation + encoder), a \theta-parametric morphism in the base category.
class WapFun where
  evalSketch :: Perm -> Op -> Bool -> Op -> Numbers -> Maybe Answer
  repF :: Weights -> Problems -> Rep

-- | KlFun -- the neural Kleisli symbols, per monad (class over @m@). 'repS' is the
--   canonical \eta-lift of the trunk symbol 'repF' (a CERTAIN Kleisli step), defined ONCE
--   here as the default: binding it in the formula shares the encoder forward across the
--   four heads; instances supply only the heads.
class (WapFun, Monad m) => WapKlFun m where
  repS :: Weights -> Problems -> m Rep
  repS theta = return . repF theta
  permuteS :: Weights -> Rep -> m Perm
  op1S :: Weights -> Rep -> m Op
  swapS :: Weights -> Rep -> m Bool
  op2S :: Weights -> Rep -> m Op
