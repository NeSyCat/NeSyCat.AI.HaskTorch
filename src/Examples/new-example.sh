#!/usr/bin/env bash
# Scaffold a new NeSyCat example: copies the full A–G template into
# src/Examples/<Name>/, renames the placeholder, and registers it in the
# dispatcher. After this, `hpack` (run automatically) picks up the new modules
# with no manual cabal edit. Usage:  src/Examples/new-example.sh <Name>   (UpperCamelCase)
set -euo pipefail

NAME="${1:-}"
if ! [[ "$NAME" =~ ^[A-Z][A-Za-z0-9]*$ ]]; then
  echo "usage: src/Examples/new-example.sh <Name>   (UpperCamelCase, e.g. SudokuSolver)" >&2
  exit 1
fi

# The script lives in src/Examples/; the repo root is two levels up.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/src/Examples/$NAME"
[ -e "$DEST" ] && { echo "error: $DEST already exists" >&2; exit 1; }

# lower-camel (template -> fooBar) and kebab (FooBar -> foo-bar) forms
LOWER="$(awk '{print tolower(substr($0,1,1)) substr($0,2)}' <<<"$NAME")"
KEBAB="$(sed -E 's/([a-z0-9])([A-Z])/\1-\2/g' <<<"$NAME" | tr '[:upper:]' '[:lower:]')"

cp -R "$ROOT/src/Examples/_template" "$DEST"
find "$DEST" -name '*.hs' -print0 |
  NAME="$NAME" LOWER="$LOWER" xargs -0 perl -i -pe 's/Template/$ENV{NAME}/g; s/template/$ENV{LOWER}/g;'

# Register in the dispatcher (insert before the markers so they persist).
NAME="$NAME" KEBAB="$KEBAB" perl -i -pe '
  s{^-- NEW-EXAMPLE-IMPORT}{import Examples.$ENV{NAME}.Example ($ENV{NAME})\n-- NEW-EXAMPLE-IMPORT};
  s{^  -- NEW-EXAMPLE-ARM}{  ("$ENV{KEBAB}" : rest) -> runExample \@$ENV{NAME} (parseN rest)\n  -- NEW-EXAMPLE-ARM};
' "$ROOT/src/Lib/Run.hs"

# Regenerate the cabal so the new modules are built (hpack globs src/).
( cd "$ROOT" && ~/.cabal/bin/hpack >/dev/null 2>&1 || hpack >/dev/null 2>&1 || \
  echo "note: run hpack manually (not found on PATH)" >&2 )

echo "Created src/Examples/$NAME/ (full A-G stack) and registered '$KEBAB'."
echo "Build + run the stub:   cabal run nesycat -- $KEBAB 1"
echo "Then fill in:           src/Examples/$NAME/{C_Domain,D_Grammatical}/*, G_Data/Loader.hs, E_Inferential.hs, F_Statistical.hs"
