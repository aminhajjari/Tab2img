#!/bin/bash

#SBATCH --job-name=baseline_comparison
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baseline_%A_%a.out
#SBATCH --error=logs/baseline_%A_%a.err

# Create logs directory
mkdir -p logs
mkdir -p baseline_results

# Dataset paths - adjust these to your dataset locations
DATASETS=(
    "datasets/balance-scale/balance-scale.csv"
    "datasets/diabetes/diabetes.csv"
    "datasets/iris/iris.csv"
    # Add more dataset paths here
)

# Activate your Python environment
# source /path/to/your/venv/bin/activate

echo "========================================"
echo "BASELINE MODELS COMPARISON BATCH RUN"
echo "========================================"
echo "Start time: $(date)"
echo "Number of datasets: ${#DATASETS[@]}"
echo ""

# Run comparison for each dataset
for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Processing: $dataset"
    echo "----------------------------------------"
    
    # Check if dataset exists
    if [ ! -f "$dataset" ]; then
        echo "WARNING: Dataset not found: $dataset"
        echo "Skipping..."
        continue
    fi
    
    # Run baseline comparison
    python baseline_models_comparison.py \
        --data "$dataset" \
        --output_dir baseline_results \
        --skip_tuning
    
    echo "Completed: $dataset"
    echo ""
done

echo "========================================"
echo "BATCH RUN COMPLETE"
echo "End time: $(date)"
echo "========================================"

# Generate aggregate report
python aggregate_baseline_results.py --results_dir baseline_results