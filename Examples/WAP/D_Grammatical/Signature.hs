{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Grammatical layer (D) -- SIGNATURE for WAP: the single abstract per-problem formula,
--   monad-polymorphic over @m@, INCLUDING its quantifier.
--
--     forall (s, (ns, y)) in data.  y = evalSketch(permute(s), op1(s), swap(s), op2(s), ns)
--
--   Faithful to the MNIST pattern: the trunk is a CERTAIN Kleisli step (@r <- repS s@, the
--   \eta-lift of a deterministic morphism -- bound once, shared by the four heads), and the
--   observation -- the problem's three numbers with its answer -- enters as @\eta (ns, y) ::
--   m (Numbers, Answer)@, bound @(ns, y) <- obs@ exactly like the sketch decisions;
--   'evalSketch' and @(==)@ are plain host ops on the bound values. Written ONCE; only the
--   monad @m@ changes. The bind supplies the marginalization (the law of total probability
--   for @Dist@; the joint mask readout for @LogTens@ -- the sketch is non-separable, so the
--   engine's exact fallback applies over the tiny 192 x |support| joint).
module WAP.D_Grammatical.Signature
  ( wapFormula,
    wapSentence,
  )
where

import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import B_Logical.Signature.Guard (Guard)
import B_Logical.Signature.TwoMonBLat (TwoMonBLat (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import WAP.C_Domain.Signature (Answer, Numbers, Omega, Problems, WapFun (..), WapKlFun (..))

-- | The per-problem FORMULA: the trunk bound once, four sketch decisions, the observation
--   bound like the decisions, then the plain function symbol 'evalSketch' compared.
wapFormula ::
  forall m.
  (WapFun, WapKlFun m) =>
  Weights ->
  (Problems, m (Numbers, Answer)) ->
  m Omega
wapFormula theta (s, obs) = do
  r <- repS @m theta s
  p <- permuteS @m theta r
  o1 <- op1S @m theta r
  w <- swapS @m theta r
  o2 <- op2S @m theta r
  (ns, y) <- obs
  return (evalSketch p o1 w o2 ns == Just y)

-- | The SENTENCE: the whole quantified axiom, abstract over @m@. The guard is the data
--   quantified over (a list of pairs in @Dist@, the batched pair in @LogTens@); the
--   universal is 'bigWedge'.
wapSentence ::
  forall m.
  ( WapFun,
    WapKlFun m,
    TwoMonBLat Omega,
    A2MonBLat m Omega,
    Monad m
  ) =>
  ParamsLogic Omega ->
  Guard m (Problems, m (Numbers, Answer)) ->
  Weights ->
  m Omega
wapSentence lp guard theta = bigWedge lp guard (wapFormula @m theta)
