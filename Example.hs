{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | A trainable + benchmarkable example: everything the generic runner needs,
--   all drawn from the example's interpretations. Adding a new example is one
--   'Example' instance wiring its interpretations together; 'runExample' then
--   trains theta* AND benchmarks it, with no per-example driver code. This is
--   the "pick an example, click train" button.
module Example (Example (..), runExample) where

import Data.Kind (Type)
import E_Inferential.Train (train)
import F_Statistical.Report (BenchmarkReport, runAverage)
import Torch (Parameterized, Randomizable)
import qualified Torch

class Example e where
  type Params e :: Type -- model parameters theta
  type Spec e :: Type   -- how to sample theta
  type Data e :: Type   -- a dataset in the fixed (tensor) format

  name :: String                                   -- shown in the report
  spec :: Spec e                                   -- parameter spec
  trainConfig :: (Int, Float)                      -- (epochs, learning rate)
  loadData :: IO (Data e)                          -- data, already in the right format
  objective :: Data e -> Params e -> Torch.Tensor  -- the loss "diagram" (axiom + losses)
  report :: Params e -> Data e -> IO BenchmarkReport -- predict + label + metrics

-- | The button: train theta* via the inferential objective, then benchmark,
--   averaging over @n@ runs (n = 1 prints the loss curve).
runExample ::
  forall e.
  (Example e, Randomizable (Spec e) (Params e), Parameterized (Params e)) =>
  Int ->
  IO ()
runExample n = do
  dat <- loadData @e
  let (epochs, lr) = trainConfig @e
  runAverage (name @e) n $ do
    theta <- train (n == 1) (spec @e) (objective @e dat) epochs lr
    report @e theta dat
