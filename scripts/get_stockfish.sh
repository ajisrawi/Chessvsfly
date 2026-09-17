#!/usr/bin/env bash
# Download a Stockfish binary (used only to label training positions).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tools && cd tools
curl -sSL -o sf.tar "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar"
tar -xf sf.tar && rm sf.tar
chmod +x stockfish/stockfish-ubuntu-x86-64-avx2
echo "export STOCKFISH=$(pwd)/stockfish/stockfish-ubuntu-x86-64-avx2"
