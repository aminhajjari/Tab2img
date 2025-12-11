#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - 80 Datasets with Weight Decay + Interpretability
#=======================================================================
# Enhanced with:
# - Weight Decay (1e-4) robustness
# - Dual SHAP interpretability (9 files per dataset)
# - Adaptive timeouts for large datasets
# - Centralized interpretability output
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_VIF_PROD
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/production_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/production_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2IMG_DIR="$PROJECT_DIR/Tab2img"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"

# Scripts
BATCH_SCRIPT="$TAB2IMG_DIR/run_all_datasets.py"  # 🆕 Use updated script
MAIN_SCRIPT="$TAB2IMG_DIR/run_vif.py"

# Output
RESULTS_BASE="$TAB2IMG_DIR/results"
JOB_LOGS_DIR="$TAB2IMG_DIR/job_logs"

# Timeout configuration (based on your experience)
TIMEOUT_DEFAULT=7200  # 2 hours (safe for most datasets)

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE-VIF PRODUCTION RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Datasets: 80 datasets"
echo "Configuration:"
echo "  - Weight Decay: 1e-4 (AdamW)"
echo "  - Dual SHAP Interpretability: Enabled"
echo "  - Timeout: 2 hours per dataset"
echo "  - CPUs: 8 cores"
echo "  - Memory: 64GB"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
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
# Verify Files
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found: $BATCH_SCRIPT"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found: $MAIN_SCRIPT"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
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

echo "Python environment:"
python --version
python -c "
import torch, shap
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'SHAP: {shap.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready"
echo ""

#=======================================================================
# Verify Weight Decay in Code
#=======================================================================
echo "Verifying weight decay configuration..."
if grep -q "weight_decay=1e-4" "$MAIN_SCRIPT"; then
    echo "✅ Weight decay (1e-4) confirmed in run_vif.py"
else
    echo "⚠️  WARNING: weight_decay not found in run_vif.py"
    echo "   Make sure it's configured correctly!"
fi
echo ""

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo "=========================================="
echo "🚀 STARTING BATCH PROCESSING"
echo "=========================================="
echo "Using updated run_all_datasets.py with:"
echo "  ✅ Centralized interpretability (--interp_root)"
echo "  ✅ SHAP file validation (9 files/dataset)"
echo "  ✅ Weight decay documentation"
echo "  ✅ Enhanced statistics"
echo ""
echo "Running command:"
echo "python $BATCH_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID \\"
echo "  --script_path $MAIN_SCRIPT \\"
echo "  --timeout $TIMEOUT_DEFAULT"
echo ""
echo "=========================================="
echo ""

# Run the batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT"

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "PRODUCTION RUN COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    # Find result directory
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS!"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📊 Files generated:"
    echo "    ├── csv/"
    echo "    │   ├── results_summary.csv"
    echo "    │   ├── statistics.csv"
    echo "    │   └── interpretability_summary.csv  ← 🆕"
    echo "    ├── latex/"
    echo "    │   └── results_latex.txt"
    echo "    ├── logs/"
    echo "    │   └── results.jsonl"
    echo "    └── interpretability/                ← 🆕"
    echo "        ├── balance-scale/dual_shap_interpretability/"
    echo "        ├── tic-tac-toe/dual_shap_interpretability/"
    echo "        └── ... (80 datasets total)"
    echo ""
    
    # Count interpretability files
    if [ -d "$RESULT_DIR/interpretability" ]; then
        INTERP_COUNT=$(find "$RESULT_DIR/interpretability" -type d -name "dual_shap_interpretability" | wc -l)
        echo "🔍 Interpretability outputs: $INTERP_COUNT/80 datasets"
        echo ""
    fi
    
    # Show quick stats if CSV exists
    if [ -f "$RESULT_DIR/csv/statistics.csv" ]; then
        echo "📊 Quick Statistics:"
        grep "Average Accuracy" "$RESULT_DIR/csv/statistics.csv" | head -1
        grep "Datasets >90%" "$RESULT_DIR/csv/statistics.csv" | head -1
        echo ""
    fi
    
    echo "📧 Completion email sent to: aminhajjr@gmail.com"
    echo ""
    echo "🎉 All 80 datasets processed with Weight Decay + Interpretability!"
    
else
    echo "⚠️  Some datasets may have failed"
    echo ""
    echo "Check logs:"
    echo "    Output: $JOB_LOGS_DIR/production_${SLURM_JOB_ID}.out"
    echo "    Error:  $JOB_LOGS_DIR/production_${SLURM_JOB_ID}.err"
    echo ""
    echo "Partial results may still be available"
fi

echo "=========================================="
exit $EXIT_CODE
