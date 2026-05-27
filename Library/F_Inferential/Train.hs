{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Generic gradient-descent training, shared by every example.
--
--   A domain supplies only an objective @params -> Tensor@ (a scalar loss,
--   closing over its own data, axiom and model forward); the Adam loop here is
--   completely domain-agnostic. This is the "train button": give it an
--   initializer (draw theta_0) and an objective, get back the optimized theta*.
module F_Inferential.Train
  ( train,
    trainBatched,
    foldLoop,
  )
where

import Control.Monad (when)
import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Text.Printf (printf)
import Torch (Parameterized (..))
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))
import Torch.NN ()
import Torch.Optim (mkAdam, runStep)

-- | Minimize a full-batch @objective@ (theta -> scalar loss) over the parameters
--   drawn by @mkInit@ using Adam for @numEpochs@ steps. A thin wrapper over
--   'trainBatched' with a single batch (the whole dataset, closed over by @objective@),
--   kept so full-batch examples need no batching machinery. @verbose@ prints the curve.
train ::
  forall params.
  (Parameterized params) =>
  Bool ->                       -- verbose (print the loss curve)
  IO params ->                  -- initializer: draw the initial theta_0
  (params -> Torch.Tensor) ->   -- objective: theta -> scalar loss
  Int ->                        -- epochs
  Float ->                      -- learning rate
  IO params
train verbose mkInit objective numEpochs learningRate =
  trainBatched verbose mkInit numEpochs learningRate
    (\_ _ -> [()]) () (\_ _ model -> objective model)

-- | Generic mini-batched, epoch-aware Adam training, shared by every example.
--
--   @mkBatches epoch dat@ slices the dataset into that epoch's batches (one batch =
--   the whole dataset recovers full-batch training); @objective epoch batch theta@ is
--   the scalar loss of one batch, with the @epoch@ available so an example can schedule
--   a parameter (e.g. the @p@ of a p-mean aggregator). Adam state is threaded across
--   every batch of every epoch, so per-batch stepping is ordinary mini-batch SGD.
trainBatched ::
  forall params dat batch.
  (Parameterized params) =>
  Bool ->                                     -- verbose (print the loss curve)
  IO params ->                                -- initializer: draw the initial theta_0
  Int ->                                      -- epochs
  Float ->                                    -- learning rate
  (Int -> dat -> [batch]) ->                  -- epoch -> dataset -> batches
  dat ->                                      -- the dataset
  (Int -> batch -> params -> Torch.Tensor) -> -- epoch -> batch -> theta -> scalar loss
  IO params
trainBatched verbose mkInit numEpochs learningRate mkBatches dat objective = do
  initModel <- mkInit
  let initOpt = mkAdam 0 0.9 0.999 (flattenParameters initModel)
      lrTens = Torch.toDevice (Device CPU 0) (Torch.asTensor learningRate)
      printEvery = max 1 (numEpochs `div` 20)
  startTime <- getCurrentTime
  (final, _) <- foldLoop (initModel, initOpt) [0 .. numEpochs - 1] $ \(m0, o0) epoch -> do
    let theBatches = mkBatches epoch dat
    (m1, o1, lastLoss) <-
      foldLoop (m0, o0, Torch.asTensor (0.0 :: Float)) theBatches $ \(m, o, _) b -> do
        let !loss = objective epoch b m
        (m', o') <- runStep m o loss lrTens
        return (m', o', loss)
    when (verbose && ((epoch + 1) `mod` printEvery == 0 || epoch == 0 || epoch == numEpochs - 1)) $ do
      epochEnd <- getCurrentTime
      let diffMs = (realToFrac (diffUTCTime epochEnd startTime) :: Double) * 1000
      putStrLn $
        printf "[Epoch %3d/%d] J=%7.5f | %.2fms"
          (epoch + 1) numEpochs (Torch.asValue lastLoss :: Float) diffMs
    return (m1, o1)
  totalEnd <- getCurrentTime
  let totalDiff = realToFrac (diffUTCTime totalEnd startTime) :: Double
  when verbose $ putStrLn (printf "[Training complete] %.2fs" totalDiff)
  return final

foldLoop :: a -> [b] -> (a -> b -> IO a) -> IO a
foldLoop acc [] _ = return acc
foldLoop acc (x : xs) f = f acc x >>= \a -> foldLoop a xs f
