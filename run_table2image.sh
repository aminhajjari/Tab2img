#!/bin/bash

#=======================================================================
# SLURM BATCH SCRIPT - Process ALL OpenML Datasets
#=======================================================================
# Processes 67+ datasets automatically with Table2Image-VIF
# Author: Amin (aminhajjr@gmail.com)
# Updated: January 2025
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/job_logs/batch_all_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/job_logs/batch_all_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2MG_DIR="$PROJECT_DIR/Tab2mg"

# Input/Output paths
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
OUTPUT_DIR="$PROJECT_DIR/ALL_RESULTS_${SLURM_JOB_ID}"
MNIST_ROOT="$PROJECT_DIR/datasets"

# Script paths
BATCH_SCRIPT="$TAB2MG_DIR/run_all_datasets.py"
MAIN_SCRIPT="$TAB2MG_DIR/run_vif.py"

# Virtual environment
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"

# Training parameters
EPOCHS=50
BATCH_SIZE=64
TIMEOUT=7200  # 2 hours per dataset

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE BATCH PROCESSING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

#=======================================================================
# GPU Information
#=======================================================================
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

#=======================================================================
# Create Directories
#=======================================================================
echo "Setting up directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MNIST_ROOT"
mkdir -p "$PROJECT_DIR/job_logs"

echo "✅ Directories created"

#=======================================================================
# Verify Paths
#=======================================================================
echo ""
echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets directory not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found: $BATCH_SCRIPT"
    echo ""
    echo "Please create this file. It should be in:"
    echo "  $TAB2MG_DIR/run_all_datasets.py"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found: $MAIN_SCRIPT"
    echo ""
    echo "Expected: $MAIN_SCRIPT"
    echo ""
    echo "Available scripts in Tab2mg:"
    ls -la "$TAB2MG_DIR"/*.py
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found: $VENV_PATH"
    exit 1
fi

# Count datasets
DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo ""
echo "✅ All paths verified"
echo "   Found $DATASET_COUNT dataset folders"

#=======================================================================
# Load Environment
#=======================================================================
echo ""
echo "Loading modules..."
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

echo ""
echo "Activating virtual environment..."
source "$VENV_PATH"

echo ""
echo "Python environment:"
echo "  Python: $(which python)"
python --version

echo ""
echo "Checking PyTorch..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: PyTorch check failed!"
    exit 1
fi

echo "✅ Environment ready"

#=======================================================================
# Show Dataset Preview
#=======================================================================
echo ""
echo "=========================================="
echo "Dataset folders to process:"
find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | head -15 | while read dir; do
    folder_name=$(basename "$dir")
    file_count=$(find "$dir" -type f \( -name "*.csv" -o -name "*.arff" -o -name "*.data" \) | wc -l)
    echo "  • $folder_name ($file_count file(s))"
done

if [ $DATASET_COUNT -gt 15 ]; then
    echo "  ... and $((DATASET_COUNT - 15)) more datasets"
fi
echo "=========================================="

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo ""
echo "🚀 STARTING BATCH PROCESSING"
echo "=========================================="
echo "Configuration:"
echo "  Datasets directory: $DATASETS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs per dataset: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Timeout: $((TIMEOUT / 3600))h per dataset"
echo ""
echo "Estimated completion:"
echo "  Best case: ~$((DATASET_COUNT / 2)) hours"
echo "  Worst case: ~$((DATASET_COUNT * 2)) hours"
echo "=========================================="
echo ""

# Run the batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --script_path "$MAIN_SCRIPT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_root "$MNIST_ROOT" \
    --timeout $TIMEOUT \
    --skip_existing

EXIT_CODE=$?

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS!"
    echo ""
    echo "📂 Results saved to:"
    echo "   $OUTPUT_DIR/"
    echo ""
    echo "📊 Key files:"
    echo "   results_summary.csv      - All dataset results"
    echo "   results_latex.txt        - LaTeX table (top 20)"
    echo "   comparison_table.txt     - Comparison template"
    echo "   progress_log.jsonl       - Execution log"
    echo "   models/                  - Trained models (.pt files)"
    echo ""
    echo "📈 To view average accuracy:"
    echo "   cat $OUTPUT_DIR/results_summary.csv | tail -1"
    echo ""
    echo "💾 To download all results:"
    echo "   scp -r shahab33@narval.alliancecan.ca:$OUTPUT_DIR/ ."
    echo ""
    echo "📧 Results summary will be emailed to: aminhajjr@gmail.com"
else
    echo "❌ FAILED (exit code: $EXIT_CODE)"
    echo ""
    echo "Check error log:"
    echo "   cat /project/def-arashmoh/shahab33/Msc/job_logs/batch_all_${SLURM_JOB_ID}.err"
    echo ""
    echo "Partial results may be in:"
    echo "   $OUTPUT_DIR/"
    echo ""
    echo "To re-run only failed datasets, use --skip_existing flag"
fi

echo "=========================================="
exit $EXIT_CODE
