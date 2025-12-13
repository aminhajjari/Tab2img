#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - VIF_base Experiment (VIF_base.py)
#=======================================================================
# UPDATED to run VIF_base.py directly
# For 80 tabular datasets - VIF-Enhanced pipeline
#=======================================================================

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

#=======================================================================
# ✅ UPDATED Configuration (RUNS VIF_base.PY)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="$PROJECT_DIR/Tab2img/Baseline"          # Main working directory
DATASETS_DIR="$PROJECT_DIR/tabularDataset"            # Datasets location
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"         # Virtual environment

# VIF_base script (THIS IS WHAT RUNS)
VIF_BASE_SCRIPT="$BASELINE_DIR/VIF_base.py"

# Output directories
RESULTS_BASE="$BASELINE_DIR/t2i_vif_results"
JOB_LOGS_DIR="$BASELINE_DIR/job_logs"

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "VIF_BASE EXPERIMENT - Running VIF_base.py"
echo "=========================================="
echo "📁 Working in: $BASELINE_DIR"
echo "📁 VIF script: $VIF_BASE_SCRIPT"
echo "📁 Datasets: $DATASETS_DIR"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo "Configuration:"
echo "  - Script: VIF_base.py (VIF-enhanced pipeline)"
echo "  - GPU: A100 | CPUs: 8 cores | Memory: 64GB"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu --format=csv,noheader
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
# Verify VIF_base Script & Datasets
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$VIF_BASE_SCRIPT" ]; then
    echo "❌ ERROR: VIF_base.py not found: $VIF_BASE_SCRIPT"
    echo "💡 Save your VIF script as: $VIF_BASE_SCRIPT"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo "✅ VIF_base.py verified: $(ls -lh $VIF_BASE_SCRIPT)"
echo ""

#=======================================================================
# Load Environment (VIF + GPU)
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

echo "Environment verification (VIF + Table2Image):"
python --version
python -c "
import torch, torchvision, statsmodels, sklearn
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'Statsmodels: {statsmodels.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('✅ VIF_base.py dependencies OK')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Required packages missing!"
    exit 1
fi

echo "✅ Environment ready (VIF computation + GPU enabled)"
echo ""

#=======================================================================
# Execute VIF_base.py (PRIMARY EXECUTION)
#=======================================================================
echo "=========================================="
echo "🚀 STARTING VIF_BASE EXPERIMENT"
echo "=========================================="
echo "📁 VIF script: $VIF_BASE_SCRIPT"
echo "📁 Datasets: $DATASETS_DIR"
echo "📁 Output: $RESULTS_BASE"
echo "📁 Logs: $JOB_LOGS_DIR"
echo ""
echo "Running command:"
echo "python $VIF_BASE_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID"
echo ""
echo "=========================================="
echo ""

# Run VIF_base.py with standard arguments
cd "$BASELINE_DIR"
python "$VIF_BASE_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID"

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "VIF_BASE EXPERIMENT COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! VIF_base.py completed successfully"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULTS_BASE/"
    echo ""
    echo "📊 Expected VIF outputs:"
    echo "    ├── vif_JOBXXXX/"
    echo "    │   ├── balance-scale.pt (trained model)"
    echo "    │   ├── balance-scale/vif_summary.json"
    echo "    │   ├── summary_vif_results.csv"
    echo "    │   └── ... (80 datasets)"
    echo ""
    
    # Show recent results
    echo "📁 Latest VIF results:"
    ls -la "$RESULTS_BASE" | tail -10
    echo ""
    
    echo "🎉 VIF-enhanced experiment complete!"
    echo "📈 Ready for comparison with baselines & original T2I"
    
else
    echo "⚠️  VIF_base.py failed with exit code $EXIT_CODE"
    echo "Check logs:"
    echo "    $JOB_LOGS_DIR/vif_base_${SLURM_JOB_ID}.out"
    echo "    $JOB_LOGS_DIR/vif_base_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
