-- | Thin executable entry point; all dispatch logic lives in "Lib.Run".
module Main (main) where

import Lib.Run (runNeSyCat)
import System.Environment (getArgs)

main :: IO ()
main = getArgs >>= runNeSyCat
