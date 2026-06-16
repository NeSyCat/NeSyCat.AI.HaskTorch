{-# LANGUAGE TypeFamilies #-}

-- | The inference signature (level epsilon): the symbols that turn a
--   parameterized formula into an optimization problem.
--
--   Sort symbol:
--     Loss                            the objective values (R)
--   Function symbols:
--     lossKnow : Omega -> Loss           knowledge loss (how unsatisfied is the axiom?)
--     lossData : Omega x Omega -> Loss   data loss (prediction vs label) [optional]
--     lossComb : Loss x Loss x lambda -> Loss   combined  J = lam*J_data + (1-lam)*J_know [optional]
--
--   Categorically the inference level works in Para(C_delta) (Gavranovic 2024),
--   over the smooth category Diff (tensor spaces with differentiable maps): the
--   whole path  theta -> h_theta(x) -> lossKnow([[phi]]_theta) -> R  must be smooth,
--   since the chain rule propagates gradients through it. Data-providing maps
--   (labels, encodings) live in Tens and are constants during differentiation.
--
--   The concrete losses are a library (F_Inferential/Library); an interpretation
--   assigns each role symbol below to one of them.
module F_Inferential.InferenceSignature
  ( InferenceSignature (..),
  )
where

import Data.Kind (Type)

-- | The inference signature: the sort symbol @Loss@ together with the function
--   symbols @lossKnow@, @lossData@, @lossComb@. An interpretation (instance)
--   assigns @Loss@ a concrete sort and each function symbol a concrete morphism.
class InferenceSignature cat where
  -- | Sort symbol: the type of objective/loss values (R).
  type Loss cat :: Type

  -- | Knowledge loss: how unsatisfied is the axiom? REQUIRED for every truth object.
  lossKnow :: cat -> Loss cat

  -- | Data loss (prediction vs label): OPTIONAL. A purely knowledge-driven reading (e.g.
  --   @LogTens Bool@, whose objective is @lossKnow . sat@) leaves it at the default, which errors
  --   only if a supervised data loss is ever actually requested -- so no probability has to be
  --   materialized for a loss that is never used.
  lossData :: cat -> cat -> Loss cat
  lossData _ _ = error "lossData: no data-loss interpretation for this truth object"

  -- | Convex combination of the data and knowledge losses: OPTIONAL, defaults with 'lossData'.
  lossComb :: Loss cat -> Loss cat -> cat -> Loss cat
  lossComb _ _ _ = error "lossComb: no data-loss interpretation for this truth object"
