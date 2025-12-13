#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - Table2Image VIF (Variance Inflation Factor)
#=======================================================================
# For 80 tabular datasets - VIF-Enhanced Table2Image
# Enhanced with:
# - VIF-based weight initialization for multicollinearity
# - AdamW optimizer (weight decay 1e-4)
# - FashionMNIST + MNIST + VIF embeddings
# - Model checkpointing + comprehensive metrics
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_VIF_ENHANCED
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/t2i_vif_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/t2i_vif_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration (IDENTICAL paths)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2IMG_DIR="$PROJECT_DIR/Tab2img"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"

# VIF-Enhanced Table2Image script
VIF_SCRIPT="$TAB2IMG_DIR/table2image_vif.py"
BATCH_SCRIPT="$TAB2IMG_DIR/run_t2i_vif_batch.py"

# Output directories
RESULTS_BASE="$TAB2IMG_DIR/t2i_vif_results"
JOB_LOGS_DIR="$TAB2IMG_DIR/job_logs"

# Timeout: 4.5 hours (VIF computation adds overhead)
TIMEOUT_DEFAULT=16200

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE VIF-ENHANCED - 80 DATASETS"
echo "=========================================="
echo "CVAE + VIF Embeddings + Tabular + FashionMNIST/MNIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Datasets: 80 datasets"
echo "Configuration:"
echo "  - Model: CVAEWithTabEmbedding + VIFInitialization"
echo "  - VIF: Multicollinearity-aware weight init"
echo "  - Images: FashionMNIST (0-9) + MNIST (10-19)"
echo "  - Optimizer: AdamW (lr=0.001, wd=1e-4)"
echo "  - Timeout: 4.5h/dataset | GPU: A100"
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
# Verify Files & Datasets
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$VIF_SCRIPT" ]; then
    echo "❌ ERROR: VIF Table2Image script not found: $VIF_SCRIPT"
    echo "💡 Save your Python code as: $VIF_SCRIPT"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found: $BATCH_SCRIPT"
    echo "💡 Create run_t2i_vif_batch.py (code below)"
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
print('✅ VIF-Table2Image dependencies OK')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready (VIF computation enabled)"
echo ""

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo "=========================================="
echo "🚀 STARTING TABLE2IMAGE-VIF BATCH"
echo "=========================================="
echo "VIF-Enhanced features:"
echo "  ✅ Variance Inflation Factor embeddings"
echo "  ✅ Inverse-VIF weight initialization"
echo "  ✅ Tabular + VIF + Image supervision"
echo "  ✅ Model checkpointing per epoch"
echo ""
echo "Command:"
echo "python $BATCH_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID \\"
echo "  --script_path $VIF_SCRIPT \\"
echo "  --timeout $TIMEOUT_DEFAULT"
echo ""
echo "=========================================="
echo ""

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$VIF_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT"

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "TABLE2IMAGE-VIF PRODUCTION COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS! VIF-Table2Image completed"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📊 Per-dataset outputs:"
    echo "    ├── balance-scale/balance-scale.pt (best model)"
    echo "    ├── balance-scale/vif_summary.json"
    echo "    ├── balance-scale/performance_history.csv"
    echo "    ├── tic-tac-toe/..."
    echo "    └── ... (80 datasets)"
    echo ""
    
    # Count completed models
    COMPLETED=$(find "$RESULT_DIR" -name "*.pt" | wc -l)
    echo "✅ $COMPLETED/80 VIF models trained"
    echo ""
    
    if [ -f "$RESULT_DIR/summary_vif_results.csv" ]; then
        echo "🏆 Top 5 VIF-Table2Image performances:"
        echo "Dataset        | Acc(%) | AUC    | VIF_Mean"
        head -6 "$RESULT_DIR/summary_vif_results.csv"
        echo ""
    fi
    
    echo "🎉 VIF-enhanced Table2Image ready for comparison!"
    echo "📈 Compare with: baseline_results/ | table2image_results/"
    
else
    echo "⚠️  Some datasets may have failed (VIF computation intensive)"
    echo "Check logs:"
    echo "    $JOB_LOGS_DIR/t2i_vif_${SLURM_JOB_ID}.out"
    echo "    $JOB_LOGS_DIR/t2i_vif_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
