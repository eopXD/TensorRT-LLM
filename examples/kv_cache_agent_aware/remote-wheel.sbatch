#!/bin/bash
#SBATCH -A coreai_comparch_trtllm
#SBATCH -p batch_long
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -J cachesage-wheel
#SBATCH -o /lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/yuehtingc/cachesage-wheel-%j.log
set -x
SCRATCH=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/yuehtingc
REPO=$SCRATCH/TensorRT-LLM-cachesage
SQSH=$SCRATCH/trtllm-devel-x86.sqsh
STATUS=$SCRATCH/cachesage-wheel-$SLURM_JOB_ID.status

echo "START $(date -u) job=$SLURM_JOB_ID node=$(hostname)" | tee "$STATUS"
git -C "$REPO" log --oneline -1 | tee -a "$STATUS"

srun --mpi=none --ntasks=1 \
  --container-image="$SQSH" \
  --container-name=cachesage-wheel \
  --container-mounts="$REPO:/code/tensorrt_llm,$SCRATCH:/lustre_scratch" \
  --container-workdir=/code/tensorrt_llm \
  bash /lustre_scratch/cachesage-inbuild.sh
rc=$?

echo "BUILD_RC=$rc $(date -u)" | tee -a "$STATUS"
ls -la "$REPO/build/"tensorrt_llm-*.whl 2>&1 | tee -a "$STATUS"
echo "END $(date -u)" | tee -a "$STATUS"
