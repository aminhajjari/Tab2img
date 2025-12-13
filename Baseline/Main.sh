#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - Main Experiment (Main.py)
#=======================================================================
# UPDATED to match new Main.py arguments
# For 80 tabular datasets - Complete experiment pipeline
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=MAIN_EXPERIMENT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/main_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/main_%A.err


#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# ✅ UPDATED Configuration (RUNS MAIN.PY)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="$PROJECT_DIR/Tab2img/Baseline"          # Main working directory
DATASETS_DIR="$PROJECT_DIR/tabularDataset"            # Datasets location
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"         # Virtual environment

# Main script (THIS IS WHAT RUNS)
MAIN_SCRIPT="$BASELINE_DIR/Main.py"

# Output directories
RESULTS_BASE="$BASELINE_DIR/baseline_results"
JOB_LOGS_DIR="$BASELINE_DIR/job_logs"

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "MAIN EXPERIMENT - Running Main.py"
echo "=========================================="
echo "📁 Working in: $BASELINE_DIR"
echo "📁 Main script: $MAIN_SCRIPT"
echo "📁 Datasets: $DATASETS_DIR"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo "Configuration:"
echo "  - Script: Main.py (complete pipeline)"
echo "  - CPUs: 8 cores | Memory: 32GB"
echo "=========================================="
echo ""

#=======================================================================
# CPU/GPU Information
#=======================================================================
echo "System Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No GPU detected"
echo "CPU cores: $(nproc)"
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
# Verify Main Script & Datasets
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main.py not found: $MAIN_SCRIPT"
    echo "💡 Save your main script as: $MAIN_SCRIPT"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo "✅ Main.py verified: $(ls -lh $MAIN_SCRIPT)"
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

echo "Python environment check:"
python --version
python -c "
import numpy, pandas, sklearn, xgboost, lightgbm, torch
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'XGBoost: {xgboost.__version__}')
print(f'LightGBM: {lightgbm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ Main.py dependencies OK')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Required packages missing!"
    exit 1
fi

echo "✅ Environment ready"
echo ""

#=======================================================================
# Execute Main.py (PRIMARY EXECUTION)
#=======================================================================
echo "=========================================="
echo "🚀 STARTING MAIN EXPERIMENT"
echo "=========================================="
echo "📁 Main script: $MAIN_SCRIPT"
echo "📁 Datasets: $DATASETS_DIR"
echo "📁 Output: $RESULTS_BASE"
echo "📁 Logs: $JOB_LOGS_DIR"
echo ""
echo "Running command:"
echo "python $MAIN_SCRIPT \\"
echo "  --data_dir $DATASETS_DIR \\"
echo "  --output_dir $RESULTS_BASE \\"
echo "  --skip_tuning \\"
echo "  --random_state 42"
echo ""
echo "=========================================="
echo ""

# Run Main.py with CORRECTED arguments
cd "$BASELINE_DIR"
python "$MAIN_SCRIPT" \
    --data_dir "$DATASETS_DIR" \
    --output_dir "$RESULTS_BASE" \
    --skip_tuning \
    --random_state 42

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "MAIN EXPERIMENT COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! Main.py completed successfully"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULTS_BASE/"
    echo ""
    echo "📊 Expected outputs:"
    echo "    ├── [dataset1]/"
    echo "    │   ├── baseline_comparison.csv"
    echo "    │   ├── baseline_comparison.png"
    echo "    │   └── baseline_results.json"
    echo "    ├── [dataset2]/"
    echo "    │   ├── baseline_comparison.csv"
    echo "    │   ├── baseline_comparison.png"
    echo "    │   └── baseline_results.json"
    echo "    └── ..."
    echo ""
    
    # Show recent results
    echo "📁 Latest results:"
    echo "Processed datasets:"
    find "$RESULTS_BASE" -maxdepth 1 -type d | tail -10
    echo ""
    
    # Count successful datasets
    SUCCESS_COUNT=$(find "$RESULTS_BASE" -name "baseline_results.json" | wc -l)
    echo "✅ Successfully processed $SUCCESS_COUNT datasets"
    echo ""
    
    echo "🎉 Main experiment pipeline complete!"
    echo "📈 Ready for analysis and visualization"
    
else
    echo "⚠️  Main.py failed with exit code $EXIT_CODE"
    echo "Check logs:"
    echo "    $JOB_LOGS_DIR/main_${SLURM_JOB_ID}.out"
    echo "    $JOB_LOGS_DIR/main_${SLURM_JOB_ID}.err"
fi

echo "=========================================="

exit $EXIT_CODE
