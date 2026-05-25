#!/usr/bin/env bash
# Scaffold a new NeSyCat example: copy the full A–G template into Examples/<Name>/
# and rename the placeholder. It is auto-registered on the next ./nesycat run (the
# dispatcher Library/Run.hs is regenerated from the folder list) — no manual step.
# Usage:  Examples/new-example.sh <Name>   (UpperCamelCase)
set -euo pipefail

NAME="${1:-}"
if ! [[ "$NAME" =~ ^[A-Z][A-Za-z0-9]*$ ]]; then
  echo "usage: Examples/new-example.sh <Name>   (UpperCamelCase, e.g. SudokuSolver)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/Examples/$NAME"
[ -e "$DEST" ] && { echo "error: $DEST already exists" >&2; exit 1; }

# rename the placeholder: Template -> Name (types/modules), template -> lowerName (values)
LOWER="$(awk '{print tolower(substr($0,1,1)) substr($0,2)}' <<<"$NAME")"
cp -R "$ROOT/Examples/_template" "$DEST"
find "$DEST" -name '*.hs' -print0 |
  NAME="$NAME" LOWER="$LOWER" xargs -0 perl -i -pe 's/Template/$ENV{NAME}/g; s/template/$ENV{LOWER}/g;'

echo "Created Examples/$NAME/ (full A-G stack; every layer is a folder, A/B empty = reused from the library)."
echo "Build + run it:   ./nesycat $NAME 1     (auto-registered by folder name; no manual step)"
echo "Then fill in the STANDALONE slots:  C_Domain/*  D_Grammatical/*  E_Data/*  F_Inferential/Interpretation.hs  G_Statistical/Interpretation.hs"
echo "(A/B are template references in Definition.hs; to customize one, drop files in its empty folder and re-point the import.)"
