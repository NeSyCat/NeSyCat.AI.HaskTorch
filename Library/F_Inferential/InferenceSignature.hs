{-# LANGUAGE TypeFamilies #-}

-- | The inference signature (level epsilon): the symbols that turn a
--   parameterized formula into an optimization problem.
--
--   Sort symbol:
--     Loss                            the objective values (R)
--   Function symbols:
--     lossKnow : Omega -> Loss           knowledge loss (how unsatisfied is the axiom?)
--     lossData : Omega x Omega -> Loss   data loss (prediction vs label)
--     lossComb : Loss x Loss x lambda -> Loss   combined  J = lam*J_data + (1-lam)*J_know
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

  lossKnow :: cat -> Loss cat

  lossData :: cat -> cat -> Loss cat

  lossComb :: Loss cat -> Loss cat -> cat -> Loss cat
