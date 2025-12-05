#!/bin/bash

#=======================================================================
# SLURM BATCH SCRIPT - Process ALL OpenML Datasets
#=======================================================================
# All outputs organized in Tab2img folder structure
# Author: Amin (aminhajjr@gmail.com)
# Updated: January 2025
# MODIFIED: Model checkpoint saving is DISABLED.
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/batch_all_%A.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/batch_all_%A.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#=======================================================================
# Configuration - All paths under Tab2img/
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2IMG_DIR="$PROJECT_DIR/Tab2img"

# Datasets location (external)
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
MNIST_ROOT="$PROJECT_DIR/datasets"

# Output locations (all under Tab2img/)
JOB_LOGS_DIR="$TAB2IMG_DIR/job_logs"
RESULTS_BASE="$TAB2IMG_DIR/results"

# Script paths
BATCH_SCRIPT="$TAB2IMG_DIR/run_all_datasets.py"
MAIN_SCRIPT="$TAB2IMG_DIR/run_vif.py"

# Virtual environment
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"

# Training parameters
TIMEOUT=7200    # 2 hours per dataset

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
# Create Directory Structure
#=======================================================================
echo "Setting up directory structure..."
mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"
mkdir -p "$MNIST_ROOT"

echo "✅ Directories created:"
echo "    Job logs:  $JOB_LOGS_DIR"
echo "    Results:   $RESULTS_BASE"

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
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found: $MAIN_SCRIPT"
    echo ""
    echo "Available scripts:"
    ls -la "$TAB2IMG_DIR"/*.py
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
echo "    Found $DATASET_COUNT dataset folders"

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
python --version
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready"

#=======================================================================
# Show Dataset Preview
#=======================================================================
echo ""
echo "=========================================="
echo "Datasets to process:"
find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | head -15 | while read dir; do
    folder_name=$(basename "$dir")
    file_count=$(find "$dir" -type f \( -name "*.csv" -o -name "*.arff" -o -name "*.data" \) | wc -l)
    echo "  • $folder_name ($file_count file)"
done

if [ $DATASET_COUNT -gt 15 ]; then
    echo "  ... and $((DATASET_COUNT - 15)) more datasets"
fi
echo "=========================================="

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo ""
echo "🚀 STARTING BATCH PROCESSING (Model saving disabled)"
echo "=========================================="
echo "Output structure:"
echo "  $RESULTS_BASE/"
echo "    └── [DATE]_JOB${SLURM_JOB_ID}/"
echo "        ├── csv/      (result tables)"
echo "        ├── latex/    (paper tables)"
echo "        └── logs/     (processing logs)"
echo ""
echo "Configuration:"
echo "  Datasets: $DATASET_COUNT"
echo "  Timeout: $((TIMEOUT / 3600))h per dataset"
echo "=========================================="
echo ""

# Run the batch processor
python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
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
    # Find the results directory
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS! Model checkpoints were NOT saved."
    echo ""
    echo "📂 Results saved to:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📊 Generated files:"
    # REMOVED: models/                - Trained models (*.pt)
    echo "    csv/results_summary.csv   - Main results table"
    echo "    csv/statistics.csv        - Summary statistics"
    echo "    latex/results_latex.txt   - LaTeX table (top 20)"
    echo "    latex/comparison_table.txt - Comparison with baselines"
    echo "    logs/progress_log.jsonl   - Execution log"
    echo ""
    echo "📈 To view average accuracy:"
    echo "    cat $RESULT_DIR/csv/statistics.csv"
    echo ""
    echo "💾 To download results to your computer:"
    echo "    scp -r shahab33@narval.alliancecan.ca:$RESULT_DIR/ ."
    echo ""
    echo "📧 Job completion email sent to: aminhajjr@gmail.com"
else
    echo "❌ FAILED (exit code: $EXIT_CODE)"
    echo ""
    echo "Check logs:"
    echo "    Output: $JOB_LOGS_DIR/batch_all_${SLURM_JOB_ID}.out"
    echo "    Error:  $JOB_LOGS_DIR/batch_all_${SLURM_JOB_ID}.err"
    echo ""
    echo "Partial results may be in: $RESULTS_BASE/"
fi

echo "=========================================="
exit $EXIT_CODE
