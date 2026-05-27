{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | The contract a runnable example satisfies, assembled from its A–G layers so the
--   per-example @Definition.hs@ is a pure manifest. The inference layer is kept BLIND
--   to everything upstream: it never sees the model, the axiom-building or the logic.
--   It receives only two things —
--
--     * @sat@   : the satisfaction value, built and exported by the GRAMMATICAL layer (D)
--                 from a parameter set and a data batch (D forms the formulas + the SAT);
--     * @dat@   : the data, exported by the DATA layer (E).
--
--   The objective is then GENERIC and lives here, not in any example: @lossKnow . sat@,
--   where @lossKnow@ is fixed by the example's 'InferenceSignature' instance (the F
--   layer — which is therefore /only/ a choice of knowledge/data/combination loss).
--   So an example's F layer is just @instance InferenceSignature (Truth e)@; C gives
--   'Params'/'initParams', E gives 'loadData'/'batches', D gives 'sat', G gives 'report'.
module Example (Example (..), runExample) where

import Data.Kind (Type)
import F_Inferential.InferenceSignature (InferenceSignature (..))
import F_Inferential.Train (trainBatched)
import G_Statistical.Report (Report, runAverage)
import Torch (Parameterized)
import qualified Torch

class Example e where
  type Params e :: Type -- C: the parameter space theta
  type Data e :: Type   -- E: a dataset in the fixed format
  type Batch e :: Type  -- E: one training mini-batch (default: the whole dataset)
  type Batch e = Data e
  type Truth e :: Type  -- D: the truth object the satisfaction lives in (F's instance picks its losses)

  initParams :: IO (Params e)             -- C: draw the initial theta_0
  loadData :: IO (Data e)                  -- E: the data, in the right format
  trainConfig :: (Int, Float)              -- (epochs, learning rate)
  batches :: Int -> Data e -> [Batch e]    -- E: epoch -> dataset -> mini-batches
  sat :: Params e -> Batch e -> Truth e    -- D: the satisfaction of the axioms over a batch
  report :: Params e -> Data e -> IO Report -- G: predict + label + labeled metrics

  -- | Default: full batch — one batch per epoch, the whole dataset (needs @Batch e ~ Data e@).
  default batches :: (Batch e ~ Data e) => Int -> Data e -> [Batch e]
  batches _ d = [d]

-- | The button: train theta* by minimizing the GENERIC objective @lossKnow . sat@
--   (the only seam the inference layer has to the rest), then benchmark, averaging over
--   @n@ runs (n = 1 prints the loss curve). @name@ is the example's folder name.
runExample ::
  forall e.
  ( Example e,
    Parameterized (Params e),
    InferenceSignature (Truth e),
    Loss (Truth e) ~ Torch.Tensor
  ) =>
  String ->
  Int ->
  IO ()
runExample name n = do
  dat <- loadData @e
  let (epochs, lr) = trainConfig @e
  runAverage name n $ do
    theta <- trainBatched (n == 1) (initParams @e) epochs lr (batches @e) dat (\_ b t -> lossKnow (sat @e t b))
    report @e theta dat
