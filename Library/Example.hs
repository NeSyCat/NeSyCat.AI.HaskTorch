{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | The contract a runnable example satisfies. Each member is contributed by one
--   of the example's A–G layers, so the per-example @Definition.hs@ is a pure
--   manifest that just points each member at its layer (C: the parameter space
--   'Params' + its initializer 'initParams'; E: data; F: 'objective'/'trainConfig'
--   /'batches'; G: 'report'; D's axiom feeds F's objective). The example's name is its
--   folder name, passed to 'runExample' by the dispatcher.
--
--   Training is mini-batched and epoch-aware: 'batches' slices a dataset into the
--   batches for an epoch (the default is one batch = the whole dataset, i.e. full
--   batch), and 'objective' takes the epoch (so an example can schedule a parameter,
--   e.g. the @p@ of a p-mean aggregator). Full-batch examples ignore both.
module Example (Example (..), runExample) where

import Data.Kind (Type)
import F_Inferential.Train (trainBatched)
import G_Statistical.Report (Report, runAverage)
import Torch (Parameterized)
import qualified Torch

class Example e where
  type Params e :: Type -- the parameter space theta (C: the horizontal sort / actor object)
  type Data e :: Type   -- a dataset in the fixed format (E)
  type Batch e :: Type  -- one training mini-batch (default: the whole dataset)
  type Batch e = Data e

  initParams :: IO (Params e)                              -- C: draw the initial theta_0
  loadData :: IO (Data e)                                  -- E: the data, in the right format
  trainConfig :: (Int, Float)                              -- F: (epochs, learning rate)
  batches :: Int -> Data e -> [Batch e]                    -- F/E: epoch -> dataset -> mini-batches
  objective :: Int -> Batch e -> Params e -> Torch.Tensor  -- F: per-batch loss (epoch for scheduling)
  report :: Params e -> Data e -> IO Report                -- G: predict + label + labeled metrics

  -- | Default: full batch — one batch per epoch, the whole dataset (needs @Batch e ~ Data e@).
  default batches :: (Batch e ~ Data e) => Int -> Data e -> [Batch e]
  batches _ d = [d]

-- | The button: train theta* via the inferential objective (mini-batched, epoch-aware),
--   then benchmark, averaging over @n@ runs (n = 1 prints the loss curve). @name@ is the
--   example's folder name, supplied by the dispatcher.
runExample ::
  forall e.
  (Example e, Parameterized (Params e)) =>
  String ->
  Int ->
  IO ()
runExample name n = do
  dat <- loadData @e
  let (epochs, lr) = trainConfig @e
  runAverage name n $ do
    theta <- trainBatched (n == 1) (initParams @e) epochs lr (batches @e) dat (objective @e)
    report @e theta dat
