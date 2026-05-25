{-# LANGUAGE TypeApplications #-}

-- | nesycat dispatcher: map an example name to the generic runner. Adding an
--   example appends one import + one case arm (the markers below let
--   @scripts/new-example.sh@ do that automatically) — everything else the example
--   needs lives in its own @Examples.<Name>@ folder.
module Run (runNeSyCat) where

import Binary.Example (Binary)
import MnistAddition.Example (MnistAdd)
-- NEW-EXAMPLE-IMPORT (do not remove: scripts/new-example.sh inserts imports above this line)
import Example (runExample)
import System.Exit (die)

runNeSyCat :: [String] -> IO ()
runNeSyCat args = case args of
  ("binary" : rest) -> runExample @Binary (parseN rest)
  ("mnist-add" : rest) -> runExample @MnistAdd (parseN rest)
  -- NEW-EXAMPLE-ARM (do not remove: scripts/new-example.sh inserts arms above this line)
  _ -> die "usage: nesycat <example> [n]   (n = runs to average; n=1 prints the loss curve)"
  where
    parseN (x : _) = read x
    parseN [] = 10
