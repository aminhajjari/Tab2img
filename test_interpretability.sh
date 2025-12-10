#!/bin/bash

#=======================================================================
# TEST SLURM SCRIPT - Validate Dual SHAP Interpretability on Small Datasets
#=======================================================================
# Quick test for dual SHAP interpretability on small datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_SHAP_TEST
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_shap_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_shap_%j.err

#SBATCH --mail-user=aminhajjr@gmail.com
#SBATCH --mail-type=END,FAIL

#=======================================================================
# Configuration
#=======================================================================
PROJECT_DIR="/project/def-arashmoh/shahab33/Msc"
TAB2IMG_DIR="$PROJECT_DIR/Tab2img"
DATASETS_DIR="$PROJECT_DIR/tabularDataset"
VENV_PATH="$PROJECT_DIR/venvMsc/bin/activate"
MAIN_SCRIPT="$TAB2IMG_DIR/run_vif.py"

# Test datasets (small ones for faster debugging)
TEST_DATASETS=(
    "balance-scale"
    "tic-tac-toe"
    "blood-transfusion-service-center"
)

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "DUAL SHAP INTERPRETABILITY TEST RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Testing ${#TEST_DATASETS[@]} datasets"
echo "=========================================="
echo ""

#=======================================================================
# GPU Information
#=======================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

#=======================================================================
# Load Environment
#=======================================================================
echo "Loading environment..."
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

echo "Activating virtual environment..."
source "$VENV_PATH"

echo ""
echo "Python environment:"
python --version
python - << 'EOF'
import torch, numpy, pandas
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF

echo ""
echo "Checking SHAP installation..."
python << 'PYCHECK'
try:
    import shap
    print(f"✓ SHAP installed: version {shap.__version__}")
except ImportError:
    print("✗ SHAP not installed - script will fail!")
    print("Install with: pip install shap")
    exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: SHAP is required for dual SHAP interpretability"
    echo "Please install: pip install shap"
    exit 1
fi

echo ""
echo "Environment ready"
echo ""

#=======================================================================
# Test Each Dataset
#=======================================================================
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_TIME=0

echo "=========================================="
echo "STARTING TEST RUNS"
echo "=========================================="

for dataset in "${TEST_DATASETS[@]}"; do
    echo ""
    echo "======================================"
    echo "Testing: $dataset"
    echo "======================================"

    DATASET_PATH=$(find "$DATASETS_DIR" -type d -name "$dataset" | head -1)

    if [ -z "$DATASET_PATH" ]; then
        echo "✗ Dataset not found: $dataset"
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi

    DATA_FILE=$(find "$DATASET_PATH" -type f \( -name "*.arff" -o -name "*.csv" -o -name "*.data" \) | head -1)

    if [ -z "$DATA_FILE" ]; then
        echo "✗ No usable data file found for: $dataset"
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi

    echo "Dataset path: $DATASET_PATH"
    echo "Data file: $DATA_FILE"
    echo ""

    START_TIME=$(date +%s)

    # ✓ CORRECTED: Removed --interpretability_dir (doesn't exist)
    # Outputs will be saved to: {dataset_name}/dual_shap_interpretability/
    python "$MAIN_SCRIPT" \
        --data "$DATA_FILE" \
        --num_images 5 \
        2>&1 | tee "/tmp/test_${dataset}_${SLURM_JOB_ID}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    TOTAL_TIME=$((TOTAL_TIME + ELAPSED))

    echo ""
    echo "--------------------------------------"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ SUCCESS in ${ELAPSED}s"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Check for interpretability outputs
        # The script creates: {dataset_name}/dual_shap_interpretability/
        INTERP_DIR="$dataset/dual_shap_interpretability"
        
        if [ -d "$INTERP_DIR" ]; then
            echo ""
            echo "Dual SHAP files generated:"
            ls -lh "$INTERP_DIR" | grep -E "\.(csv|png|txt|npy)$"
            echo ""
            echo "File count:"
            echo "  CSV files: $(find "$INTERP_DIR" -name "*.csv" | wc -l)"
            echo "  PNG plots: $(find "$INTERP_DIR" -name "*.png" | wc -l)"
            echo "  NPY arrays: $(find "$INTERP_DIR" -name "*.npy" | wc -l)"
            echo "  Text reports: $(find "$INTERP_DIR" -name "*.txt" | wc -l)"
        else
            echo "⚠ Warning: Interpretability directory not found: $INTERP_DIR"
        fi

    else
        echo "✗ FAILED (exit code: $EXIT_CODE) after ${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "Error preview (last 30 lines):"
        tail -30 "/tmp/test_${dataset}_${SLURM_JOB_ID}.log"
    fi

    echo "======================================"
done


#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "DUAL SHAP TEST RESULTS SUMMARY"
echo "=========================================="
echo "Datasets tested: ${#TEST_DATASETS[@]}"
echo "✓ Success: $SUCCESS_COUNT"
echo "✗ Failed:  $FAIL_COUNT"
echo "Total time: ${TOTAL_TIME}s ($(($TOTAL_TIME/60))m)"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Interpretability outputs are in:"
    for dataset in "${TEST_DATASETS[@]}"; do
        if [ -d "$dataset/dual_shap_interpretability" ]; then
            echo "  • $dataset/dual_shap_interpretability/"
        fi
    done
fi

echo ""
echo "Finished: $(date)"
echo "=========================================="

# Exit with failure if any test failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
