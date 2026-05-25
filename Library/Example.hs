{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | The contract a runnable example satisfies. Each member is contributed by one
--   of the example's A–G layers, so the per-example @Example.hs@ is a pure
--   manifest that just points each member at its layer (C: model/'spec'; E: data;
--   F: 'objective'/'trainConfig'; G: 'report'; D's axiom feeds F's objective). The
--   example's name is its folder name, passed to 'runExample' by the dispatcher.
module Example (Example (..), runExample) where

import Data.Kind (Type)
import F_Inferential.Train (train)
import G_Statistical.Report (Report, runAverage)
import Torch (Parameterized, Randomizable)
import qualified Torch

class Example e where
  type Params e :: Type -- model parameters theta            (C: the domain's model)
  type Spec e :: Type   -- how to sample theta               (C)
  type Data e :: Type   -- a dataset in the fixed format      (E)

  spec :: Spec e                                   -- C: how to initialize theta
  loadData :: IO (Data e)                          -- E: the data, in the right format
  trainConfig :: (Int, Float)                      -- F: (epochs, learning rate)
  objective :: Data e -> Params e -> Torch.Tensor  -- F: the loss (penalty of D's axiom over the data)
  report :: Params e -> Data e -> IO Report        -- G: predict + label + labeled metrics

-- | The button: train theta* via the inferential objective, then benchmark,
--   averaging over @n@ runs (n = 1 prints the loss curve). @name@ is the
--   example's folder name, supplied by the dispatcher.
runExample ::
  forall e.
  (Example e, Randomizable (Spec e) (Params e), Parameterized (Params e)) =>
  String ->
  Int ->
  IO ()
runExample name n = do
  dat <- loadData @e
  let (epochs, lr) = trainConfig @e
  runAverage name n $ do
    theta <- train (n == 1) (spec @e) (objective @e dat) epochs lr
    report @e theta dat
