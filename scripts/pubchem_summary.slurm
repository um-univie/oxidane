#!/bin/bash
#SBATCH --job-name=oxidane-summary
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scr/kaldra/muhlir
set -euo pipefail
RUN_DIR=$1
export TMPDIR="$RUN_DIR/tmp/summary"
export UV_CACHE_DIR=/scr/kaldra/muhlir/cache/uv
mkdir -p "$TMPDIR"
uv run --no-project python "$RUN_DIR/summarize_pubchem_sweep.py" "$RUN_DIR"
