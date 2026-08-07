#!/bin/bash
#SBATCH -A coreai_comparch_trtllm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -J cachesage-verify
#SBATCH -o /lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/yuehtingc/cachesage-verify-%j.log
set -x
SCRATCH=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/yuehtingc
REPO=$SCRATCH/TensorRT-LLM-cachesage
SQSH=$SCRATCH/trtllm-devel-x86.sqsh
STATUS=$SCRATCH/cachesage-verify-$SLURM_JOB_ID.status

echo "START $(date -u) job=$SLURM_JOB_ID node=$(hostname)" | tee "$STATUS"

srun --mpi=none --ntasks=1 \
  --container-image="$SQSH" \
  --container-name=cachesage-verify \
  --container-mounts="$REPO:/code/tensorrt_llm,$SCRATCH:/lustre_scratch" \
  --container-workdir=/code/tensorrt_llm \
  bash /lustre_scratch/cachesage-verify.sh
rc=$?

echo "VERIFY_RC=$rc $(date -u)" | tee -a "$STATUS"
echo "END $(date -u)" | tee -a "$STATUS"
