{-# LANGUAGE TypeFamilies #-}

-- | WAP example -- the A-G manifest: Word Algebra Problems (Roy & Roth's Common Core set),
--   the canonical NeSy STRING benchmark (differentiable-Forth 96.0, DeepProbLog up to 96.5 /
--   re-run 94.2 +/- 1.4, DeepStochLog 94.8 +/- 1.1). The raw input sort is TEXT, not a
--   tensor: the C interpretation owns the collation (reference tokenizer -> Embedding ->
--   BiGRU), demonstrating that the framework never assumed tensor-carried sorts. The
--   observation -- the problem's numbers with its answer -- enters as @\eta (ns, y)@, a
--   one-hot leaf over the batch's distinct pairs built by the E layer (the MNIST pattern;
--   answers have unbounded global support, but a BATCH only carries finitely many).
module WAP.Definition (WAP) where

-- A (category) -- REUSE shared: the LogVec monad (the truth object's carrier).
import A_Categorical.Monads.LogVec (LogVec)
-- B (logic) -- REUSE template: Boolean (the crisp truth algebra + the Dist quantifier).
import B_Logical.Interpretations.Boolean ()
-- C (domain) -- STANDALONE: WAP's sorts/symbols + model (Params/initParams).
import qualified WAP.C_Domain.Interpretation as C
-- D (grammatical) -- STANDALONE: the axiom in both monads; 'sat' = the LogVec reading.
import qualified WAP.D_Grammatical.Interpretation as D
-- E (data) -- STANDALONE: the committed reference data + loader/batches.
import qualified WAP.E_Data.Signature as E
import qualified WAP.E_Data.Loader as EL
-- F (inference) -- REUSE template signature AND the library's shared
--   @instance InferenceSignature (LogVec Bool)@; STANDALONE only for trainConfig.
import F_Inferential.InferenceSignature ()
import F_Inferential.InferenceInterpretation ()
import qualified WAP.F_Inferential.Interpretation as F
-- G (statistics) -- REUSE template signature; STANDALONE interpretation (report).
import G_Statistical.BenchmarkSignature ()
import qualified WAP.G_Statistical.Interpretation as G
import Example (Example (..))
import WAP.C_Domain.Signature (Answer, Numbers, Problem)

data WAP

instance Example WAP where
  type Params WAP = C.Params
  type Data WAP = E.Dataset
  type Batch WAP = ([Problem], LogVec (Numbers, Answer))
  type Truth WAP = LogVec Bool
  initParams = C.initParams
  loadData = EL.loadData
  trainConfig = F.trainConfig
  batches = EL.batches
  sat = D.wapAxiomTens
  report = G.report
