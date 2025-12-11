#!/bin/bash

#=======================================================================
# TEST SLURM SCRIPT - Validate Dual SHAP + Weight Decay
#=======================================================================
# Quick test for dual SHAP interpretability + robustness on small datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_ROBUST_TEST
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --output=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_robust_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_robust_%j.err

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
echo "ROBUSTNESS + INTERPRETABILITY TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Testing ${#TEST_DATASETS[@]} datasets"
echo "Robustness: Weight Decay = 1e-4"
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

# 🆕 Arrays to store results
declare -a DATASET_NAMES
declare -a TAB_ACCURACIES
declare -a IMG_ACCURACIES

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
        
        # 🆕 Extract accuracy from log
        TAB_ACC=$(grep "Best Accuracy" "/tmp/test_${dataset}_${SLURM_JOB_ID}.log" | tail -1 | grep -oP '\d+\.\d+' | head -1)
        IMG_ACC=$(grep "Test Accuracy - Image" "/tmp/test_${dataset}_${SLURM_JOB_ID}.log" | tail -1 | grep -oP '\d+\.\d+' | head -1)
        
        if [ -n "$TAB_ACC" ]; then
            DATASET_NAMES+=("$dataset")
            TAB_ACCURACIES+=("$TAB_ACC")
            IMG_ACCURACIES+=("$IMG_ACC")
            echo "   📊 Accuracy: Tabular=$TAB_ACC%, Image=$IMG_ACC%"
        fi

        # Check for interpretability outputs
        INTERP_DIR="$dataset/dual_shap_interpretability"
        
        if [ -d "$INTERP_DIR" ]; then
            echo ""
            echo "Dual SHAP files generated:"
            CSV_COUNT=$(find "$INTERP_DIR" -name "*.csv" | wc -l)
            PNG_COUNT=$(find "$INTERP_DIR" -name "*.png" | wc -l)
            NPY_COUNT=$(find "$INTERP_DIR" -name "*.npy" | wc -l)
            TXT_COUNT=$(find "$INTERP_DIR" -name "*.txt" | wc -l)
            
            echo "  CSV files: $CSV_COUNT"
            echo "  PNG plots: $PNG_COUNT"
            echo "  NPY arrays: $NPY_COUNT"
            echo "  Text reports: $TXT_COUNT"
            
            if [ $CSV_COUNT -eq 3 ] && [ $PNG_COUNT -eq 3 ] && [ $NPY_COUNT -eq 2 ] && [ $TXT_COUNT -eq 1 ]; then
                echo "  ✓ All 9 files present"
            else
                echo "  ⚠ Expected 9 files (3 CSV, 3 PNG, 2 NPY, 1 TXT)"
            fi
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
echo "TEST RESULTS SUMMARY"
echo "=========================================="
echo "Datasets tested: ${#TEST_DATASETS[@]}"
echo "✓ Success: $SUCCESS_COUNT"
echo "✗ Failed:  $FAIL_COUNT"
echo "Total time: ${TOTAL_TIME}s ($(($TOTAL_TIME/60))m)"
echo ""

# 🆕 Accuracy Summary Table
if [ ${#DATASET_NAMES[@]} -gt 0 ]; then
    echo "=========================================="
    echo "ACCURACY RESULTS (With Weight Decay)"
    echo "=========================================="
    printf "%-40s %10s %10s\n" "Dataset" "Tabular" "Image"
    echo "----------------------------------------"
    
    for i in "${!DATASET_NAMES[@]}"; do
        printf "%-40s %9s%% %9s%%\n" "${DATASET_NAMES[$i]}" "${TAB_ACCURACIES[$i]}" "${IMG_ACCURACIES[$i]}"
    done
    echo "=========================================="
    echo ""
fi

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

# 🆕 Save results for comparison
if [ ${#DATASET_NAMES[@]} -gt 0 ]; then
    RESULTS_FILE="test_results_weight_decay_${SLURM_JOB_ID}.txt"
    {
        echo "TEST RESULTS - Weight Decay = 1e-4"
        echo "Date: $(date)"
        echo ""
        printf "%-40s %10s %10s\n" "Dataset" "Tabular" "Image"
        echo "----------------------------------------"
        for i in "${!DATASET_NAMES[@]}"; do
            printf "%-40s %9s%% %9s%%\n" "${DATASET_NAMES[$i]}" "${TAB_ACCURACIES[$i]}" "${IMG_ACCURACIES[$i]}"
        done
    } > "$RESULTS_FILE"
    
    echo ""
    echo "Results saved to: $RESULTS_FILE"
fi

# Exit with failure if any test failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
