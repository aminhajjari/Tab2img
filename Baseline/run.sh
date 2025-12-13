#!/bin/bash
#
# PRODUCTION SLURM SCRIPT - Table2Image (Baseline)
#=======================================================================
#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_BASELINE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/run_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/run_%A.err
#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="${PROJECT_DIR}/Tab2img/Baseline"
DATASETS_DIR="${PROJECT_DIR}/tabularDataset"
VENV_PATH="${PROJECT_DIR}/venvMsc/bin/activate"
JOB_LOGS_DIR="${BASELINE_DIR}/job_logs"

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE BASELINE - BATCH EXPERIMENT"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No GPU available"
echo ""

#=======================================================================
# Create Directories
#=======================================================================
echo "Setting up directories..."
mkdir -p "${JOB_LOGS_DIR}"
mkdir -p "${BASELINE_DIR}/results"
echo "✅ Directories ready"
echo ""

#=======================================================================
# Verify Paths
#=======================================================================
echo "Verifying paths..."

if [ ! -d "${DATASETS_DIR}" ]; then
    echo "❌ ERROR: Datasets directory not found: ${DATASETS_DIR}"
    exit 1
fi

if [ ! -f "${VENV_PATH}" ]; then
    echo "❌ ERROR: Virtual environment not found: ${VENV_PATH}"
    exit 1
fi

if [ ! -f "${BASELINE_DIR}/run.py" ]; then
    echo "❌ ERROR: run.py not found: ${BASELINE_DIR}/run.py"
    exit 1
fi

if [ ! -f "${BASELINE_DIR}/run_all.py" ]; then
    echo "❌ ERROR: run_all.py not found: ${BASELINE_DIR}/run_all.py"
    echo "💡 Create run_all.py using the provided script"
    exit 1
fi

DATASET_COUNT=$(find "${DATASETS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found ${DATASET_COUNT} dataset directories"
echo "✅ All required files verified"
echo ""

#=======================================================================
# Activate Virtual Environment
#=======================================================================
echo "Activating virtual environment..."
source "${VENV_PATH}"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment active"
echo ""

#=======================================================================
# Verify Python Environment
#=======================================================================
echo "Python environment check:"
echo "Python path: $(which python)"
python --version

python -c "
import torch
import numpy
import pandas
import sklearn
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('✅ All dependencies OK')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Python environment check failed"
    exit 1
fi

echo ""

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo "=========================================="
echo "🚀 STARTING BATCH PROCESSING"
echo "=========================================="
echo "Processing ${DATASET_COUNT} datasets..."
echo "This will process all datasets in:"
echo "    ${DATASETS_DIR}"
echo ""
echo "Results will be saved to:"
echo "    ${BASELINE_DIR}/results/run_${SLURM_JOB_ID}/"
echo ""
echo "Timeout per dataset: 2 hours"
echo "Estimated total time: ~160 hours (if all timeout)"
echo "Expected time: ~40-80 hours (realistic)"
echo ""
echo "=========================================="
echo ""

cd "${BASELINE_DIR}"

# Run the batch processor
python run_all.py

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    RESULT_DIR="${BASELINE_DIR}/results/run_${SLURM_JOB_ID}"
    
    echo "✅ ALL DATASETS PROCESSED SUCCESSFULLY"
    echo ""
    echo "📂 Results location:"
    echo "    ${RESULT_DIR}/"
    echo ""
    echo "📊 Files generated:"
    echo "    ├── batch_summary.json    (detailed results)"
    echo "    ├── batch_summary.csv     (simple summary)"
    echo "    └── [dataset_name]/       (80 dataset folders)"
    echo ""
    
    # Show summary statistics if available
    if [ -f "${RESULT_DIR}/batch_summary.json" ]; then
        echo "📈 Quick Statistics:"
        python -c "
import json
with open('${RESULT_DIR}/batch_summary.json') as f:
    data = json.load(f)
    print(f\"  Total: {data['total_datasets']}\")
    print(f\"  Success: {data['success']}\")
    print(f\"  Failed: {data['failed']}\")
    print(f\"  Skipped: {data['skipped']}\")
" 2>/dev/null || echo "  (Could not parse summary)"
        echo ""
    fi
    
    echo "🎉 Baseline experiment complete!"
    
else
    echo "⚠️  Some datasets may have failed"
    echo ""
    echo "Check logs:"
    echo "    Output: ${JOB_LOGS_DIR}/run_${SLURM_JOB_ID}.out"
    echo "    Error:  ${JOB_LOGS_DIR}/run_${SLURM_JOB_ID}.err"
    echo ""
    echo "📂 Partial results may be available at:"
    echo "    ${BASELINE_DIR}/results/run_${SLURM_JOB_ID}/"
fi

echo "=========================================="
exit ${EXIT_CODE}
