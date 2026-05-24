{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Generic gradient-descent training, shared by every example.
--
--   A domain supplies only an objective @params -> Tensor@ (a scalar loss,
--   closing over its own data, axiom and model forward); the Adam loop here is
--   completely domain-agnostic. This is the "train button": give it a parameter
--   spec and an objective, get back the optimized parameters theta*.
module E_Inferential.Train
  ( train,
    foldLoop,
  )
where

import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Text.Printf (printf)
import Torch (Parameterized (..), Randomizable (..))
import qualified Torch
import Torch.Device (Device (..), DeviceType (..))
import Torch.NN ()
import Torch.Optim (mkAdam, runStep)

-- | Minimize @objective@ over the parameter space described by @spec@ using Adam
--   for @numEpochs@ steps. Works for any @Randomizable@/@Parameterized@ model
--   (MLP, CNN, ...). @verbose@ prints the loss curve.
train ::
  forall spec params.
  (Randomizable spec params, Parameterized params) =>
  Bool ->                       -- verbose (print the loss curve)
  spec ->                       -- parameter spec (e.g. binarySpecReal, ParamsCNNSpec)
  (params -> Torch.Tensor) ->   -- objective: theta -> scalar loss
  Int ->                        -- epochs
  Float ->                      -- learning rate
  IO params
train verbose spec objective numEpochs learningRate = do
  initModel <- sample spec
  let initOpt = mkAdam 0 0.9 0.999 (flattenParameters initModel)
      lrTens = Torch.toDevice (Device CPU 0) (Torch.asTensor learningRate)
  startTime <- getCurrentTime
  (final, _) <- foldLoop (initModel, initOpt) [1 .. numEpochs] $ \(model, opt) epoch -> do
    let !loss = objective model
    (newModel, newOpt) <- runStep model opt loss lrTens
    if verbose && (epoch `mod` 100 == 0 || epoch == numEpochs || epoch == 1)
      then do
        epochEnd <- getCurrentTime
        let diffMs = (realToFrac (diffUTCTime epochEnd startTime) :: Double) * 1000
        putStrLn $
          printf "[Epoch %3d/%d] J=%7.5f | %.2fms"
            epoch numEpochs (Torch.asValue loss :: Float) diffMs
      else return ()
    return (newModel, newOpt)
  totalEnd <- getCurrentTime
  let totalDiff = realToFrac (diffUTCTime totalEnd startTime) :: Double
  if verbose then putStrLn (printf "[Training complete] %.2fs" totalDiff) else return ()
  return final

foldLoop :: a -> [b] -> (a -> b -> IO a) -> IO a
foldLoop acc [] _ = return acc
foldLoop acc (x : xs) f = f acc x >>= \a -> foldLoop a xs f
