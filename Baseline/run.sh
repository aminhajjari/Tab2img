#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - Table2Image (Base paper)
#=======================================================================
# UPDATED to run run.py directly
# For 80 tabular datasets - Official Table2Image implementation
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

#=======================================================================
# ✅ UPDATED Configuration (RUNS run.PY)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="$PROJECT_DIR/Tab2img/Baseline"          # Main working directory
DATASETS_DIR="$PROJECT_DIR/tabularDataset"            # Datasets location
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"         # Virtual environment

# run script (THIS IS WHAT RUNS)
RUN_SCRIPT="$BASELINE_DIR/run.py"

# Output directories
RESULTS_BASE="$BASELINE_DIR/table2image_results"
JOB_LOGS_DIR="$BASELINE_DIR/job_logs"

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE EXPERIMENT - Running run.py"
echo "=========================================="
echo "📁 Working in: $BASELINE_DIR"
echo "📁 Run script: $RUN_SCRIPT"
echo "📁 Datasets: $DATASETS_DIR"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo "Configuration:"
echo "  - Script: run.py (Table2Image pipeline)"
echo "  - Model: CVAEWithTabEmbedding (50 epochs)"
echo "  - Images: FashionMNIST (0-9) + MNIST (10-19)"
echo "  - Optimizer: AdamW (lr=0.001)"
echo "  - GPU: A100 | CPUs: 8 cores | Memory: 64GB"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""

#=======================================================================
# Setup Directories
#=======================================================================
echo "Creating directories..."
mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"
echo "✅ Directories ready"
echo ""

#=======================================================================
# Verify run Script & Datasets
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "❌ ERROR: run.py not found: $RUN_SCRIPT"
    echo "💡 Save your Table2Image script as: $RUN_SCRIPT"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo "✅ run.py verified: $(ls -lh $RUN_SCRIPT)"
echo ""

#=======================================================================
# Load Environment
#=======================================================================
echo "Loading modules..."
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
echo "✅ Modules loaded"
echo ""

echo "Activating virtual environment..."
source "$VENV_PATH"
echo "✅ Virtual environment active"
echo ""

echo "Environment verification:"
python --version
python -c "
import torch, torchvision
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('✅ Table2Image (run.py) dependencies OK')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready"
echo ""

#=======================================================================
# Execute run.py (PRIMARY EXECUTION)
#=======================================================================
echo "=========================================="
echo "🚀 STARTING TABLE2IMAGE EXPERIMENT"
echo "=========================================="
echo "📁 Run script: $RUN_SCRIPT"
echo "📁 Datasets: $DATASETS_DIR"
echo "📁 Output: $RESULTS_BASE"
echo "📁 Logs: $JOB_LOGS_DIR"
echo ""
echo "Running command:"
echo "python $RUN_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID"
echo ""
echo "=========================================="
echo ""

# Run run.py with standard arguments
cd "$BASELINE_DIR"
python "$RUN_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID"

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "TABLE2IMAGE EXPERIMENT COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! run.py completed successfully"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULTS_BASE/"
    echo ""
    echo "📊 Expected Table2Image outputs:"
    echo "    ├── t2i_JOBXXXX/"
    echo "    │   ├── balance-scale.pt (trained model)"
    echo "    │   ├── summary_t2i_results.csv"
    echo "    │   └── ... (80 datasets)"
    echo ""
    
    # Show recent results
    echo "📁 Latest Table2Image results:"
    ls -la "$RESULTS_BASE" | tail -10
    echo ""
    
    echo "🎉 Original Table2Image ready for comparison!"
    echo "📈 Compare with: results/ | t2i_vif_results/"
    
else
    echo "⚠️  run.py failed with exit code $EXIT_CODE"
    echo "Check logs:"
    echo "    $JOB_LOGS_DIR/run_${SLURM_JOB_ID}.out"
    echo "    $JOB_LOGS_DIR/run_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
