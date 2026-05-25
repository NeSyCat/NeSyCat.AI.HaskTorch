{-# LANGUAGE TypeFamilies #-}

-- | MnistAddition example — the A–G manifest. Each layer module provides its
--   piece; this file only points the 'Example' contract at them. The example's
--   name is the folder name (supplied by the dispatcher).
module MnistAddition.Example (MnistAddition) where

import MnistAddition.A_Categorical ()                        -- A: category (universes)
import MnistAddition.B_Logical ()                            -- B: logic
import qualified MnistAddition.C_Domain.Interpretation as C  -- C: domain + model (Params/Spec/spec)
import MnistAddition.D_Grammatical.InterpretationTens ()     -- D: the sum-logits term (consumed by F)
import qualified MnistAddition.E_Data.Loader as E            -- E: data (Dataset/loadData)
import qualified MnistAddition.F_Inferential as F            -- F: objective + trainConfig
import qualified MnistAddition.G_Statistical as G            -- G: report
import Example (Example (..))

data MnistAddition

instance Example MnistAddition where
  type Params MnistAddition = C.Params
  type Spec MnistAddition = C.Spec
  type Data MnistAddition = E.Dataset
  spec = C.spec
  loadData = E.loadData
  trainConfig = F.trainConfig
  objective = F.objective
  report = G.report
