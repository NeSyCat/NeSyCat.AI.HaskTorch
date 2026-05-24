{-# LANGUAGE TypeApplications #-}

-- | nesycat: pick a domain, train its parameters, and score it.
--
--   Usage:  cabal run nesycat -- <binary|mnist-add> [n]
--     n = number of runs to average (default 10; n=1 prints the loss curve).
--
--   Each domain is one 'Domain' instance; this dispatcher just maps a name to
--   the generic 'runDomain' for that instance.
module Main where

import Example (runExample)
import Examples.Binary (Binary)
import Examples.MnistAddition (MnistAdd)
import System.Environment (getArgs)
import System.Exit (die)

main :: IO ()
main = do
  args <- getArgs
  case args of
    ("binary" : rest) -> runExample @Binary (parseN rest)
    ("mnist-add" : rest) -> runExample @MnistAdd (parseN rest)
    _ -> die "usage: nesycat <binary|mnist-add> [n]"
  where
    parseN (x : _) = read x
    parseN [] = 10
