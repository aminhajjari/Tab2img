#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - Table2Image (CVAE with Tabular Embeddings)
#=======================================================================
# For 80 tabular datasets - Official Table2Image implementation
# Enhanced with:
# - AdamW optimizer (weight decay 1e-4)
# - FashionMNIST + MNIST image supervision
# - Synchronized tabular-image datasets
# - 50 epochs, GPU-accelerated
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_ORIGINAL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/table2image_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/table2image_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration (SAME paths as your previous scripts)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2IMG_DIR="$PROJECT_DIR/Tab2img"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"

# Table2Image script (SAVE your Python code as this file)
T2I_SCRIPT="$TAB2IMG_DIR/table2image_original.py"
BATCH_SCRIPT="$TAB2IMG_DIR/run_t2i_batch.py"  # Batch processor

# Output directories
RESULTS_BASE="$TAB2IMG_DIR/table2image_results"
JOB_LOGS_DIR="$TAB2IMG_DIR/job_logs"

# Timeout: 4 hours per dataset (Table2Image is slower due to image processing)
TIMEOUT_DEFAULT=14400

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE ORIGINAL - 80 DATASETS"
echo "=========================================="
echo "CVAE + Tabular Embeddings + FashionMNIST/MNIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Datasets: 80 datasets"
echo "Configuration:"
echo "  - Model: CVAEWithTabEmbedding (50 epochs)"
echo "  - Images: FashionMNIST (0-9) + MNIST (10-19)"
echo "  - Optimizer: AdamW (lr=0.001)"
echo "  - Timeout: 4 hours/dataset"
echo "  - GPU: A100 | CPUs: 8 | Memory: 64GB"
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
# Verify Files & Datasets
#=======================================================================
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$T2I_SCRIPT" ]; then
    echo "❌ ERROR: Table2Image script not found: $T2I_SCRIPT"
    echo "💡 Save your Python code as: $T2I_SCRIPT"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found: $BATCH_SCRIPT"
    echo "💡 Create run_t2i_batch.py (code below)"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo ""

#=======================================================================
# Load Environment (IDENTICAL to your previous script)
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
print('✅ Table2Image dependencies OK')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready"
echo ""

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo "=========================================="
echo "🚀 STARTING TABLE2IMAGE BATCH PROCESSING"
echo "=========================================="
echo "Command:"
echo "python $BATCH_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID \\"
echo "  --script_path $T2I_SCRIPT \\"
echo "  --timeout $TIMEOUT_DEFAULT"
echo ""
echo "=========================================="
echo ""

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$T2I_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT"

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "TABLE2IMAGE PRODUCTION RUN COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS! Table2Image completed"
    echo ""
    echo "📂 Results location:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📊 Per-dataset outputs:"
    echo "    ├── balance-scale.pt (trained model)"
    echo "    ├── balance-scale/  (detailed logs)"
    echo "    ├── tic-tac-toe.pt"
    echo "    └── ... (80 datasets)"
    echo ""
    
    # Count completed models
    COMPLETED=$(find "$RESULT_DIR" -name "*.pt" | wc -l)
    echo "✅ $COMPLETED/80 models trained"
    echo ""
    
    if [ -f "$RESULT_DIR/summary_t2i_results.csv" ]; then
        echo "📊 Top 5 Table2Image performances:"
        head -6 "$RESULT_DIR/summary_t2i_results.csv"
        echo ""
    fi
    
    echo "🎉 Table2Image ready for baseline comparison!"
    
else
    echo "⚠️  Some datasets may have failed"
    echo "Check logs:"
    echo "    $JOB_LOGS_DIR/table2image_${SLURM_JOB_ID}.out"
    echo "    $JOB_LOGS_DIR/table2image_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
