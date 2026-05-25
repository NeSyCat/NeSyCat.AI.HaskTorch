{-# LANGUAGE TypeFamilies #-}

-- | Template example — the A–G manifest. Each layer module provides its piece;
--   this file only points the 'Example' contract at them (the name is the folder
--   name, supplied by the dispatcher). This stub builds and runs as-is
--   (@nesycat <name> 1@); fill in the C/D/E/F/G layers to make it a real experiment.
module Template.Example (Template) where

import Template.A_Categorical ()                        -- A: category (universes)
import Template.B_Logical ()                            -- B: logic
import qualified Template.C_Domain.Interpretation as C  -- C: domain + model (Params/Spec/spec)
import Template.D_Grammatical.InterpretationTens ()     -- D: the axiom (consumed by F.objective)
import qualified Template.E_Data.Loader as E            -- E: data (Dataset/loadData)
import qualified Template.F_Inferential as F            -- F: objective + trainConfig
import qualified Template.G_Statistical as G            -- G: report
import Example (Example (..))

data Template

instance Example Template where
  type Params Template = C.Params
  type Spec Template = C.Spec
  type Data Template = E.Dataset
  spec = C.spec
  loadData = E.loadData
  trainConfig = F.trainConfig
  objective = F.objective
  report = G.report
