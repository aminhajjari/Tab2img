#!/bin/bash

#=======================================================================
# TEST SLURM SCRIPT - Validate Interpretability on Small Datasets
#=======================================================================
# Quick test to verify interpretability works before running full batch
# Author: Amin (aminhajjr@gmail.com)
# Purpose: Test gradient-based interpretability on 3 small datasets
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

# Test datasets (small and fast)
TEST_DATASETS=(
    "balance-scale"         # 625 samples, 4 features, 3 classes
    "tic-tac-toe"          # 958 samples, 9 features, 2 classes
    "blood-transfusion-service-center"  # 748 samples, 4 features, 2 classes
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
echo "Testing ${#TEST_DATASETS[@]} small datasets"
echo "Purpose: Validate interpretability implementation"
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
python -c "
import torch
import numpy
import pandas
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Check if SHAP is installed
echo ""
echo "Checking SHAP installation..."
python << 'PYCHECK'
try:
    import shap
    print(f'✅ SHAP installed: version {shap.__version__}')
    SHAP_AVAILABLE=True
except ImportError:
    print('⚠️  SHAP not installed - will use gradient-based method')
    SHAP_AVAILABLE=False
PYCHECK

echo ""
echo "✅ Environment ready"
echo ""

#=======================================================================
# Test Each Dataset
#=======================================================================
echo "=========================================="
echo "STARTING TEST RUNS"
echo "=========================================="
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for dataset in "${TEST_DATASETS[@]}"; do
    echo ""
    echo "======================================"
    echo "Testing: $dataset"
    echo "======================================"
    
    # Find dataset path
    DATASET_PATH=$(find "$DATASETS_DIR" -type d -name "$dataset" | head -1)
    
    if [ -z "$DATASET_PATH" ]; then
        echo "❌ Dataset not found: $dataset"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Find data file
    DATA_FILE=$(find "$DATASET_PATH" -type f \( -name "*.arff" -o -name "*.csv" -o -name "*.data" \) | head -1)
    
    if [ -z "$DATA_FILE" ]; then
        echo "❌ No data file found in: $DATASET_PATH"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    echo "Dataset path: $DATASET_PATH"
    echo "Data file: $(basename $DATA_FILE)"
    echo ""
    echo "Running Table2Image with interpretability..."
    echo ""
    
    # Run with reduced epochs for speed
    START_TIME=$(date +%s)
    
    python "$MAIN_SCRIPT" \
        --data "$DATA_FILE" \
        --num_images 5 \
        2>&1 | tee "/tmp/test_${dataset}_${SLURM_JOB_ID}.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo "--------------------------------------"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS in ${ELAPSED}s"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Check if interpretability files were created
        INTERP_DIR="/project/def-arashmoh/shahab33/Msc/Tab2img/imageout/$dataset/interpretability"
        
        if [ -d "$INTERP_DIR" ]; then
            echo ""
            echo "📊 Interpretability files:"
            ls -lh "$INTERP_DIR"
            
            # Check specific files
            if [ -f "$INTERP_DIR/dualshap_scores.csv" ]; then
                echo "  ✅ dualshap_scores.csv"
            fi
            if [ -f "$INTERP_DIR/feature_importance.png" ]; then
                echo "  ✅ feature_importance.png"
            fi
            if [ -f "$INTERP_DIR/summary.txt" ]; then
                echo "  ✅ summary.txt"
                echo ""
                echo "Summary preview:"
                head -15 "$INTERP_DIR/summary.txt"
            fi
        else
            echo "⚠️  Interpretability directory not found: $INTERP_DIR"
            echo "   This might indicate interpretability calculation failed"
        fi
        
    else
        echo "❌ FAILED (exit code: $EXIT_CODE) in ${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        
        # Show error preview
        echo ""
        echo "Error preview from log:"
        tail -20 "/tmp/test_${dataset}_${SLURM_JOB_ID}.log"
    fi
    
    echo "======================================"
    echo ""
done

#=======================================================================
# Final Summary
#=======================================================================
echo ""
echo "=========================================="
echo "TEST RESULTS SUMMARY"
echo "=========================================="
echo "Total datasets tested: ${#TEST_DATASETS[@]}"
echo "  ✅ Success: $SUCCESS_COUNT"
echo "  ❌ Failed:  $FAIL_COUNT"
echo ""

if [ $SUCCESS_COUNT -eq ${#TEST_DATASETS[@]} ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo ""
    echo "Your interpretability implementation is working correctly."
    echo "You can now run the full batch with confidence:"
    echo "  sbatch batch_all_optimized.sh"
    echo ""
    EXIT_STATUS=0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  PARTIAL SUCCESS"
    echo ""
    echo "Some tests passed, but $FAIL_COUNT failed."
    echo "Review the errors above before running full batch."
    echo ""
    EXIT_STATUS=1
else
    echo "❌ ALL TESTS FAILED"
    echo ""
    echo "Please fix the errors before running full batch."
    echo "Common issues:"
    echo "  1. Missing SHAP (use gradient-based method)"
    echo "  2. Syntax error in run_vif.py"
    echo "  3. Missing dependencies"
    echo ""
    EXIT_STATUS=1
fi

echo "Finished: $(date)"
echo "=========================================="
echo ""

# Show where to find results
echo "Test outputs saved to:"
echo "  Job log:    /project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_interp_${SLURM_JOB_ID}.out"
echo "  Error log:  /project/def-arashmoh/shahab33/Msc/Tab2img/job_logs/test_interp_${SLURM_JOB_ID}.err"
echo "  Results:    /project/def-arashmoh/shahab33/Msc/Tab2img/imageout/"
echo ""

exit $EXIT_STATUS
