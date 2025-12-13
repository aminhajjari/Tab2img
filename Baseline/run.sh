#!/bin/bash

#
# PRODUCTION SLURM SCRIPT - Table2Image (Baseline)
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_RUN
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/run_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/run_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#
# PROJECT DIR
#
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="${PROJECT_DIR}/Tab2img/Baseline"
JOB_LOGS_DIR="${BASELINE_DIR}/job_logs"

echo "=========================================="
echo "TABLE2IMAGE BASELINE EXPERIMENT"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "=========================================="

# Create directories
mkdir -p "${JOB_LOGS_DIR}"
mkdir -p "${BASELINE_DIR}/results"

# Activate virtual environment
echo "Activating venvMsc..."
source "${BASELINE_DIR}/venvMsc/bin/activate"

# Verify environment
echo "Python path: $(which python)"
python --version

# Set paths
DATA_DIR="${BASELINE_DIR}/tabularDataset"
SAVE_DIR="${BASELINE_DIR}/results/run_${SLURM_JOB_ID}"

# Create save directory
mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "DATA DIR: ${DATA_DIR}"
echo "SAVE DIR: ${SAVE_DIR}"
echo "=========================================="

# Verify paths exist
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ ERROR: Data directory not found: ${DATA_DIR}"
    echo "Check if tabularDataset/ exists with your 80 OpenML datasets"
    exit 1
fi

cd "${BASELINE_DIR}"

# Run Table2Image
echo "🚀 Starting Table2Image pipeline..."
echo "Command: python run.py --data ${DATA_DIR} --save_dir ${SAVE_DIR}"

python run.py \
    --data "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}"

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ TABLE2IMAGE EXPERIMENT COMPLETE"
    echo "📁 Results saved to: ${SAVE_DIR}"
    echo "📋 Logs available at:"
    echo "    ${JOB_LOGS_DIR}/run_${SLURM_JOB_ID}.out"
    echo "    ${JOB_LOGS_DIR}/run_${SLURM_JOB_ID}.err"
else
    echo "⚠️  run.py failed with exit code ${EXIT_CODE}"
    echo "Check logs:"
    echo "    ${JOB_LOGS_DIR}/run_${SLURM_JOB_ID}.out"
    echo "    ${JOB_LOGS_DIR}/run_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit ${EXIT_CODE}
