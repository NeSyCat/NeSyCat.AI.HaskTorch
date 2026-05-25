-- | Thin executable entry point; all dispatch logic lives in "Run".
module Main (main) where

import Run (runNeSyCat)
import System.Environment (getArgs)

main :: IO ()
main = getArgs >>= runNeSyCat
