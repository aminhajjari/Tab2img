#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - Baseline Models Comparison (XGBoost/LightGBM/PyTorch MLP)
#=======================================================================
# UPDATED for NEW Baseline directory structure
# For 80 tabular datasets - compares against Table2Image baseline
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=BASELINE_COMP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/baseline_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/baseline_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# ✅ UPDATED Configuration (NEW BASELINE DIRECTORY)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="$PROJECT_DIR/Tab2img/Baseline"          # 🆕 NEW BASE DIR
DATASETS_DIR="$PROJECT_DIR/tabularDataset"            # ✅ SAME
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"         # ✅ SAME

# Baseline scripts (in Baseline directory)
BASELINE_SCRIPT="$BASELINE_DIR/baseline_comparison.py"
BATCH_SCRIPT="$BASELINE_DIR/run_baseline_batch.py"

# Output directories (INSIDE Baseline folder)
RESULTS_BASE="$BASELINE_DIR/results"
JOB_LOGS_DIR="$BASELINE_DIR/job_logs"

# Timeout configuration
TIMEOUT_DEFAULT=7200  # 2 hours per dataset (faster than Table2Image)

#=======================================================================
# Job Information (UPDATED PATHS)
#=======================================================================
echo "=========================================="
echo "BASELINE MODELS COMPARISON - 80 DATASETS"
echo "=========================================="
echo "📁 Working in: $BASELINE_DIR"
echo "📁 Datasets:  $DATASETS_DIR"
echo "XGBoost | LightGBM | PyTorch MLP"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo "Configuration:"
echo "  - Models: XGBoost, LightGBM, PyTorch MLP"
echo "  - Hyperparameter tuning: Enabled"
echo "  - Timeout: 2 hours per dataset"
echo "  - CPUs: 8 cores | Memory: 32GB"
echo "=========================================="
echo ""

#=======================================================================
# CPU/GPU Information
#=======================================================================
echo "System Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No GPU detected (CPU-only fine for baselines)"
echo "CPU cores: $(nproc)"
echo ""

#=======================================================================
# Setup
#=======================================================================
echo "Creating directories..."
mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"
echo "✅ Directories ready"
echo ""

#=======================================================================
# Verify Files & Datasets (UPDATED PATHS)
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BASELINE_SCRIPT" ]; then
    echo "❌ ERROR: Baseline script not found: $BASELINE_SCRIPT"
    echo "💡 Save your Python code as: $BASELINE_SCRIPT"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found: $BATCH_SCRIPT"
    echo "💡 Create run_baseline_batch.py in: $BASELINE_DIR/"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo ""

#=======================================================================
# Load Environment (SAME as Table2Image)
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
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Required packages missing!"
    exit 1
fi

echo "✅ Environment ready"
echo ""

#=======================================================================
# Execute Batch Processing (UPDATED PATHS)
#=======================================================================
echo "=========================================="
echo "🚀 STARTING BASELINE COMPARISON"
echo "=========================================="
echo "📁 Scripts: $BASELINE_DIR/"
echo "📁 Output:  $RESULTS_BASE/"
echo "Command:"
echo "python $BATCH_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID \\"
echo "  --script_path $BASELINE_SCRIPT \\"
echo "  --timeout $TIMEOUT_DEFAULT \\"
echo "  --skip_tuning False"
echo ""
echo "=========================================="
echo ""

# Run batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$BASELINE_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT" \
    --skip_tuning False  # Enable tuning for best results

EXIT_CODE=$?

#=======================================================================
# Final Summary (UPDATED PATHS)
#=======================================================================
echo ""
echo "=========================================="
echo "BASELINE COMPARISON COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    # Find result directory
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS! All baselines completed"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📊 Directory structure:"
    echo "    Baseline/"
    echo "    ├── results/baseline_JOBXXXX/"
    echo "    │   ├── balance-scale/baseline_comparison.csv"
    echo "    │   ├── balance-scale/baseline_comparison.png"
    echo "    │   └── ... (80 datasets)"
    echo "    ├── table2image_results/   (T2I original)"
    echo "    ├── t2i_vif_results/       (VIF T2I)"
    echo "    └── job_logs/"
    echo ""
    
    # Count completed datasets
    COMPLETED=$(find "$RESULT_DIR" -name "baseline_comparison.csv" 2>/dev/null | wc -l)
    echo "✅ $COMPLETED/80 datasets completed"
    echo ""
    
    # Show top performers if summary exists
    if [ -f "$RESULT_DIR/summary_all_baselines.csv" ]; then
        echo "🏆 Top 5 datasets by XGBoost accuracy:"
        head -6 "$RESULT_DIR/summary_all_baselines.csv"
        echo ""
    fi
    
    echo "🎉 Baselines ready for Table2Image comparison!"
    echo "📈 Compare with: table2image_results/ | t2i_vif_results/"
    
else
    echo "⚠️  Some datasets may have failed"
    echo "Check: $JOB_LOGS_DIR/baseline_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
