#!/bin/bash
# One shard of a PubChem sweep per array task, on one core. Shard the corpus
# finely, so that no task is long and the array spreads over the cluster:
#
#   split -n l/1000 -d -a 3 CID-SMILES SHARD_DIR/shard_
#   cp target/release/examples/pubchem_sweep RUN_DIR/ && (cd RUN_DIR && sha256sum pubchem_sweep > binary.sha256)
#   sbatch --output=RUN_DIR/logs/%a.out --error=RUN_DIR/logs/%a.err \
#       scripts/pubchem_sweep.slurm RUN_DIR SHARD_DIR [sweep options, e.g. --parent on]
#
# The shards and the binary are read over the network, so each task copies
# both to its node's own disk once and works from there, and the tasks start
# staggered across a minute rather than all at once; the `%` throttle below
# bounds how many read at a time. Raise it on a running array with
# `scontrol update JobId=... ArrayTaskThrottle=...`.
#SBATCH --job-name=oxidane-sweep
#SBATCH --partition=main
#SBATCH --array=0-999%128
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
set -euo pipefail
RUN_DIR=$1
SHARD_DIR=$2
shift 2
SHARD=$(printf 'shard_%03d' "$SLURM_ARRAY_TASK_ID")
sleep $((SLURM_ARRAY_TASK_ID % 60))
LOCAL_ROOT=/scr/$(hostname -s)/$USER
[ -d "$LOCAL_ROOT" ] || LOCAL_ROOT=${SLURM_TMPDIR:-$RUN_DIR/tmp}
WORK="$LOCAL_ROOT/oxidane-sweep-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
cp "$RUN_DIR/pubchem_sweep" "$RUN_DIR/binary.sha256" "$WORK/"
cp "$SHARD_DIR/$SHARD" "$WORK/shard"
cd "$WORK"
sha256sum --check --quiet binary.sha256
EXPECTED=$(wc -l < shard)
echo "SHARD_START shard=$SHARD expected=$EXPECTED host=$(hostname -s)"
set +e
./pubchem_sweep shard --threads 1 "$@"
STATUS=$?
set -e
echo "SHARD_COMPLETE shard=$SHARD expected=$EXPECTED exit=$STATUS"
exit "$STATUS"
