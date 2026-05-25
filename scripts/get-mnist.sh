#!/usr/bin/env bash
# Download the MNIST IDX files (gzipped) into the MnistAddition example folder
# (src/Examples/MnistAddition/data/) — example-specific data lives with its example.
# Usage:  scripts/get-mnist.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/src/Examples/MnistAddition/data"
mkdir -p "$DIR"

# PyTorch's S3 mirror (the canonical yann.lecun.com host is frequently down).
BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

for f in "${FILES[@]}"; do
  if [ -s "$DIR/$f" ]; then
    echo "have   $f"
  else
    echo "fetch  $f"
    curl -fsSL "$BASE/$f" -o "$DIR/$f"
  fi
done

echo "MNIST ready in $DIR"
