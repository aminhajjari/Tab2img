#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=VIF_BASE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/vif_base_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/vif_base_%A.err
#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="${PROJECT_DIR}/Tab2img/Baseline"
DATASETS_DIR="${PROJECT_DIR}/tabularDataset"  # FIXED
VENV_PATH="${PROJECT_DIR}/venvMsc/bin/activate"  # FIXED
JOB_LOGS_DIR="${BASELINE_DIR}/job_logs"

echo "=========================================="
echo "TABLE2IMAGE VIF_BASE BATCH EXPERIMENT"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "=========================================="

mkdir -p "${JOB_LOGS_DIR}"
mkdir -p "${BASELINE_DIR}/results"

echo "Activating venvMsc..."
source "${VENV_PATH}"  # FIXED

echo "Python path: $(which python)"
python --version

SAVE_DIR="${BASELINE_DIR}/results/vif_base_${SLURM_JOB_ID}"
mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "DATASETS DIR: ${DATASETS_DIR}"  # FIXED
echo "SAVE DIR: ${SAVE_DIR}"
echo "=========================================="

if [ ! -d "${DATASETS_DIR}" ]; then
    echo "❌ ERROR: Data directory not found: ${DATASETS_DIR}"
    exit 1
fi

cd "${BASELINE_DIR}"

# Use batch runner
echo "🚀 Starting VIF_base batch processing..."
python run_all_vif.py  # Use batch wrapper

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="
exit ${EXIT_CODE}
