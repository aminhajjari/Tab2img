#!/bin/bash

#=======================================================================
# TEST SLURM SCRIPT - Validate Interpretability on Small Datasets
#=======================================================================
# Quick test for gradient-based interpretability on small datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_TEST
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_interp_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_interp_%j.err

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

# *** New interpretability root folder ***
INTERP_ROOT="$TAB2IMG_DIR/interpretability_results"

mkdir -p "$INTERP_ROOT"

# Test datasets
TEST_DATASETS=(
    "balance-scale"
    "tic-tac-toe"
    "blood-transfusion-service-center"
)

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "INTERPRETABILITY TEST RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Testing ${#TEST_DATASETS[@]} datasets"
echo "Interpretability root: $INTERP_ROOT"
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
    print(f"SHAP installed: version {shap.__version__}")
except ImportError:
    print("SHAP not installed - gradient-based interpretability will be used")
PYCHECK

echo ""
echo "Environment ready"
echo ""

#=======================================================================
# Test Each Dataset
#=======================================================================
SUCCESS_COUNT=0
FAIL_COUNT=0

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
        echo "Dataset not found: $dataset"
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi

    DATA_FILE=$(find "$DATASET_PATH" -type f \( -name "*.arff" -o -name "*.csv" -o -name "*.data" \) | head -1)

    if [ -z "$DATA_FILE" ]; then
        echo "No usable data file found for: $dataset"
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi

    echo "Dataset path: $DATASET_PATH"
    echo "Data file: $DATA_FILE"
    echo ""

    # Create dataset-specific interpretability output folder
    INTERP_OUT="$INTERP_ROOT/$dataset"
    mkdir -p "$INTERP_OUT"
    echo "Interpretability output folder: $INTERP_OUT"
    echo ""

    START_TIME=$(date +%s)

    python "$MAIN_SCRIPT" \
        --data "$DATA_FILE" \
        --num_images 5 \
        --interpretability_dir "$INTERP_OUT" \
        2>&1 | tee "/tmp/test_${dataset}_${SLURM_JOB_ID}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "--------------------------------------"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS in ${ELAPSED}s"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        if [ -d "$INTERP_OUT" ]; then
            echo "Files in $INTERP_OUT:"
            ls -lh "$INTERP_OUT"
        else
            echo "Interpretability directory not found: $INTERP_OUT"
        fi

    else
        echo "FAILED (exit code: $EXIT_CODE)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "Error preview:"
        tail -20 "/tmp/test_${dataset}_${SLURM_JOB_ID}.log"
    fi

    echo "======================================"
done


#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "TEST RESULTS SUMMARY"
echo "=========================================="
echo "Tested: ${#TEST_DATASETS[@]}"
echo "Success: $SUCCESS_COUNT"
echo "Failed:  $FAIL_COUNT"
echo ""
echo "Interpretability outputs are stored in:"
echo "    $INTERP_ROOT"
echo ""
echo "Finished: $(date)"
echo "=========================================="

exit 0
