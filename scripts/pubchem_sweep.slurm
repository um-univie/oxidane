#!/bin/bash
#SBATCH --job-name=oxidane-normalize
#SBATCH --partition=main
#SBATCH --array=0-499%160
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scr/kaldra/muhlir
set -euo pipefail
RUN_DIR=$1
SHARD_DIR=$2
export TMPDIR="$RUN_DIR/tmp/$SLURM_ARRAY_TASK_ID"
mkdir -p "$TMPDIR"
cd "$RUN_DIR"
sha256sum --check binary.sha256
SHARD=$(printf 'shard_%03d' "$SLURM_ARRAY_TASK_ID")
EXPECTED=$(wc -l < "$SHARD_DIR/$SHARD")
echo "SHARD_START shard=$SHARD expected=$EXPECTED host=$(hostname -s)"
set +e
./pubchem_sweep "$SHARD_DIR/$SHARD" --threads 1 --max-atoms 0 --cache on
STATUS=$?
set -e
echo "SHARD_COMPLETE shard=$SHARD expected=$EXPECTED exit=$STATUS"
exit "$STATUS"
