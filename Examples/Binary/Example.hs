{-# LANGUAGE TypeFamilies #-}

-- | Binary example — the A–G manifest. Each layer module provides its piece; this
--   file only points the 'Example' contract at them. The example's name is the
--   folder name (supplied by the dispatcher), so it isn't specified here.
module Binary.Example (Binary) where

import Binary.A_Categorical ()                        -- A: category (universes)
import Binary.B_Logical ()                            -- B: logic
import qualified Binary.C_Domain.Interpretation as C  -- C: domain + model (Params/Spec/spec)
import Binary.D_Grammatical.InterpretationTens ()     -- D: the axiom (consumed by F.objective)
import qualified Binary.E_Data.Loader as E            -- E: data (Dataset/loadData)
import qualified Binary.F_Inferential as F            -- F: objective + trainConfig
import qualified Binary.G_Statistical as G            -- G: report
import Example (Example (..))

data Binary

instance Example Binary where
  type Params Binary = C.Params
  type Spec Binary = C.Spec
  type Data Binary = E.Dataset
  spec = C.spec
  loadData = E.loadData
  trainConfig = F.trainConfig
  objective = F.objective
  report = G.report
