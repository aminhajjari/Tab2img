#!/bin/bash

#=======================================================================
# PRODUCTION SLURM SCRIPT - Table2Image (Base paper)
#=======================================================================
# UPDATED for NEW Baseline directory structure
# For 80 tabular datasets - Official Table2Image implementation
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_ORIGINAL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/table2image_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline/job_logs/table2image_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# ✅ UPDATED Configuration (NEW BASELINE DIRECTORY)
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
BASELINE_DIR="$PROJECT_DIR/Tab2img/Baseline"          # 🆕 NEW BASE DIR
DATASETS_DIR="$PROJECT_DIR/tabularDataset"            # ✅ SAME
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"         # ✅ SAME

# Table2Image scripts (in Baseline directory)
T2I_SCRIPT="$BASELINE_DIR/table2image_original.py"
BATCH_SCRIPT="$BASELINE_DIR/run_t2i_batch.py"

# Output directories (INSIDE Baseline folder)
RESULTS_BASE="$BASELINE_DIR/table2image_results"
JOB_LOGS_DIR="$BASELINE_DIR/job_logs"

# Timeout: 4 hours per dataset
TIMEOUT_DEFAULT=14400

#=======================================================================
# Job Information (UPDATED PATHS)
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE ORIGINAL - 80 DATASETS"
echo "=========================================="
echo "📁 Working in: $BASELINE_DIR"
echo "📁 Datasets:  $DATASETS_DIR"
echo "CVAE + Tabular Embeddings + FashionMNIST/MNIST"
echo "Job ID: $SLURM_JOB_ID | Started: $(date)"
echo "Configuration:"
echo "  - Model: CVAEWithTabEmbedding (50 epochs)"
echo "  - Images: FashionMNIST (0-9) + MNIST (10-19)"
echo "  - Optimizer: AdamW (lr=0.001)"
echo "  - Timeout: 4 hours/dataset | GPU: A100"
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
# Verify Files & Datasets (UPDATED PATHS)
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
    echo "💡 Create run_t2i_batch.py in: $BASELINE_DIR/"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"
echo ""

#=======================================================================
# Load Environment (UNCHANGED)
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
# Execute Batch Processing (UPDATED PATHS)
#=======================================================================
echo "=========================================="
echo "🚀 STARTING TABLE2IMAGE BATCH PROCESSING"
echo "=========================================="
echo "📁 Scripts: $BASELINE_DIR/"
echo "📁 Output:  $RESULTS_BASE/"
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
# Final Summary (UPDATED PATHS)
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
    echo "📊 Directory structure:"
    echo "    Baseline/"
    echo "    ├── table2image_results/t2i_JOBXXXX/"
    echo "    │   ├── balance-scale.pt"
    echo "    │   ├── summary_t2i_results.csv"
    echo "    │   └── ... (80 datasets)"
    echo "    ├── results/          (baselines)"
    echo "    └── job_logs/"
    echo ""
    
    # Count completed models
    COMPLETED=$(find "$RESULT_DIR" -name "*.pt" 2>/dev/null | wc -l)
    echo "✅ $COMPLETED/80 models trained"
    echo ""
    
    if [ -f "$RESULT_DIR/summary_t2i_results.csv" ]; then
        echo "📊 Top 5 Table2Image performances:"
        head -6 "$RESULT_DIR/summary_t2i_results.csv"
        echo ""
    fi
    
    echo "🎉 Table2Image ready alongside baselines!"
    
else
    echo "⚠️  Some datasets may have failed"
    echo "Check logs:"
    echo "    $JOB_LOGS_DIR/table2image_${SLURM_JOB_ID}.out"
    echo "    $JOB_LOGS_DIR/table2image_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
