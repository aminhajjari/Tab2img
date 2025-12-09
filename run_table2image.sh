#!/bin/bash

#=======================================================================
# SLURM BATCH SCRIPT - Process ALL OpenML Datasets (Timeout-Optimized)
#=======================================================================
# Intelligently handles large datasets and prevents timeouts
# Author: Amin (aminhajjr@gmail.com)
# Updated: January 2025
# OPTIMIZED: Dynamic timeout, large dataset detection, adaptive settings
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2I_ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

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

#=======================================================================
# TIMEOUT CONFIGURATION (Tiered approach)
#=======================================================================
# Small datasets (< 10k samples)
TIMEOUT_SMALL=1800      # 30 minutes

# Medium datasets (10k - 50k samples)
TIMEOUT_MEDIUM=3600     # 1 hour

# Large datasets (50k - 100k samples)
TIMEOUT_LARGE=7200      # 2 hours

# Very large datasets (> 100k samples)
TIMEOUT_XLARGE=14400    # 4 hours

# Default timeout for unknown sizes
TIMEOUT_DEFAULT=5400    # 1.5 hours

#=======================================================================
# DATASETS TO SKIP (Known issues)
#=======================================================================
SKIP_DATASETS=(
    "letter"              # >20 classes
    "isolet"              # >20 classes
    "Devnagari-Script"    # >20 classes
    "MNIST-784"           # Used for mapping
    "Fashion-MNIST"       # Used for mapping
    "audiology"           # >20 classes
    "100-plants"          # >20 classes
    "guillermo"           # Only 1 class - invalid
)

#=======================================================================
# LARGE DATASETS (Need optimized settings)
#=======================================================================
LARGE_DATASETS=(
    "CIFAR"               # 60k samples
    "poker-hand"          # 1M samples
    "artificial-characters" # Large
    "connect-4"           # 67k samples
)

#=======================================================================
# Job Information
#=======================================================================
echo "=========================================="
echo "TABLE2IMAGE BATCH PROCESSING (OPTIMIZED)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Datasets: 80 (excluding 8 invalid)"
echo "Timeout Strategy: Adaptive (30m - 4h)"
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
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found: $MAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found: $VENV_PATH"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✅ Found $DATASET_COUNT dataset folders"

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
# Create Optimized run_all_datasets.py with adaptive timeout
#=======================================================================
cat > "${TAB2IMG_DIR}/run_all_datasets_adaptive.py" << 'PYTHON_EOF'
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from scipy.io import arff
import pandas as pd
import numpy as np

# Configuration from environment
DATASETS_DIR = os.environ.get('DATASETS_DIR')
RESULTS_BASE = os.environ.get('RESULTS_BASE')
MAIN_SCRIPT = os.environ.get('MAIN_SCRIPT')
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

# Timeout configuration
TIMEOUT_SMALL = int(os.environ.get('TIMEOUT_SMALL', 1800))
TIMEOUT_MEDIUM = int(os.environ.get('TIMEOUT_MEDIUM', 3600))
TIMEOUT_LARGE = int(os.environ.get('TIMEOUT_LARGE', 7200))
TIMEOUT_XLARGE = int(os.environ.get('TIMEOUT_XLARGE', 14400))
TIMEOUT_DEFAULT = int(os.environ.get('TIMEOUT_DEFAULT', 5400))

# Skip and large dataset lists
SKIP_DATASETS = os.environ.get('SKIP_DATASETS', '').split(',')
LARGE_DATASETS = os.environ.get('LARGE_DATASETS', '').split(',')

def get_dataset_size(dataset_path):
    """Get number of samples in dataset"""
    try:
        for file in os.listdir(dataset_path):
            if file.endswith('.arff'):
                arff_path = os.path.join(dataset_path, file)
                data, meta = arff.loadarff(arff_path)
                df = pd.DataFrame(data)
                return len(df)
            elif file.endswith('.csv'):
                csv_path = os.path.join(dataset_path, file)
                df = pd.read_csv(csv_path)
                return len(df)
    except Exception as e:
        print(f"Warning: Could not determine size: {e}")
    return None

def get_adaptive_timeout(dataset_name, dataset_path):
    """Determine timeout based on dataset size"""
    
    # Check if it's a known large dataset
    if dataset_name in LARGE_DATASETS:
        return TIMEOUT_XLARGE
    
    # Get actual size
    n_samples = get_dataset_size(dataset_path)
    
    if n_samples is None:
        return TIMEOUT_DEFAULT
    
    # Adaptive timeout based on size
    if n_samples < 10000:
        return TIMEOUT_SMALL
    elif n_samples < 50000:
        return TIMEOUT_MEDIUM
    elif n_samples < 100000:
        return TIMEOUT_LARGE
    else:
        return TIMEOUT_XLARGE

def get_optimized_epochs(dataset_name, n_samples):
    """Determine optimal epochs based on dataset size"""
    
    if n_samples is None:
        return 100  # Default
    
    if n_samples < 5000:
        return 100
    elif n_samples < 20000:
        return 75
    elif n_samples < 50000:
        return 50
    else:
        return 30  # Large datasets

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(RESULTS_BASE) / f"{timestamp}_JOB{SLURM_JOB_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all datasets
    datasets = sorted([d for d in os.listdir(DATASETS_DIR) 
                      if os.path.isdir(os.path.join(DATASETS_DIR, d))
                      and d not in SKIP_DATASETS])
    
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(datasets)} DATASETS WITH ADAPTIVE TIMEOUTS")
    print(f"{'='*70}\n")
    
    results = []
    success_count = 0
    fail_count = 0
    timeout_count = 0
    skip_count = 0
    
    start_time = time.time()
    
    for idx, dataset in enumerate(datasets, 1):
        dataset_path = os.path.join(DATASETS_DIR, dataset)
        
        # Get adaptive settings
        n_samples = get_dataset_size(dataset_path)
        timeout = get_adaptive_timeout(dataset, dataset_path)
        epochs = get_optimized_epochs(dataset, n_samples)
        
        # Determine batch size based on dataset size
        if n_samples and n_samples > 50000:
            batch_size = 256
        elif n_samples and n_samples > 20000:
            batch_size = 128
        else:
            batch_size = 64
        
        print(f"\n{'='*70}")
        print(f"Processing: {dataset}")
        print(f"{'='*70}")
        print(f"Folder: {dataset}")
        
        if n_samples:
            print(f"Samples: {n_samples:,}")
            print(f"Optimized settings:")
            print(f"  - Timeout: {timeout//60}m")
            print(f"  - Epochs: {epochs}")
            print(f"  - Batch size: {batch_size}")
        else:
            print(f"Using default settings (timeout: {timeout//60}m)")
        
        # Find data file
        data_files = [f for f in os.listdir(dataset_path) 
                     if f.endswith(('.arff', '.csv', '.data'))]
        
        if not data_files:
            print(f"⏭️  SKIPPED: No data file found")
            skip_count += 1
            continue
        
        data_file = data_files[0]
        
        # Build command with optimized parameters
        cmd = [
            'python', MAIN_SCRIPT,
            '--dataset', dataset,
            '--data_folder', dataset_path,
            '--arff_file', data_file,
            '--output_dir', str(output_dir),
            '--image_dir', f'/project/def-arashmoh/shahab33/Msc/Tab2img/imageout/{dataset}',
            '--save_model', 'false',
            '--epochs', str(epochs),
            '--batch_size', str(batch_size)
        ]
        
        # Run with timeout
        dataset_start = time.time()
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - dataset_start
            
            if result.returncode == 0:
                print(f"✅ SUCCESS in {elapsed:.1f}s")
                success_count += 1
            else:
                print(f"❌ FAILED (exit code {result.returncode})")
                print(f"Error: {result.stderr[-500:]}")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - dataset_start
            print(f"⏱️  TIMEOUT after {elapsed:.1f}s ({timeout}s limit)")
            timeout_count += 1
        
        # Progress
        elapsed_total = (time.time() - start_time) / 3600
        remaining = (len(datasets) - idx) * (elapsed_total / idx)
        print(f"\n{'='*70}")
        print(f"Progress: {idx}/{len(datasets)} datasets")
        print(f"Elapsed: {elapsed_total:.1f}h | Remaining: ~{remaining:.1f}h")
        print(f"✅ {success_count} | ❌ {fail_count} | ⏱️ {timeout_count}")
        print(f"{'='*70}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"  ⏱️  Timeout: {timeout_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"{'='*70}\n")
    
    # Return appropriate exit code
    if success_count > 0:
        return 0  # Success if any datasets processed
    else:
        return 1  # Fail only if nothing succeeded

if __name__ == '__main__':
    sys.exit(main())
PYTHON_EOF

#=======================================================================
# Export Configuration
#=======================================================================
export DATASETS_DIR
export RESULTS_BASE
export MAIN_SCRIPT
export SLURM_JOB_ID
export TIMEOUT_SMALL
export TIMEOUT_MEDIUM
export TIMEOUT_LARGE
export TIMEOUT_XLARGE
export TIMEOUT_DEFAULT
export SKIP_DATASETS=$(IFS=,; echo "${SKIP_DATASETS[*]}")
export LARGE_DATASETS=$(IFS=,; echo "${LARGE_DATASETS[*]}")

#=======================================================================
# Execute Batch Processing
#=======================================================================
echo ""
echo "🚀 STARTING ADAPTIVE BATCH PROCESSING"
echo "=========================================="
echo "Strategy:"
echo "  Small datasets (<10k):    30 min timeout, 100 epochs"
echo "  Medium datasets (10-50k): 1 hour timeout, 75 epochs"
echo "  Large datasets (50-100k): 2 hours timeout, 50 epochs"
echo "  XLarge datasets (>100k):  4 hours timeout, 30 epochs"
echo ""
echo "Known large datasets:"
for ds in "${LARGE_DATASETS[@]}"; do
    echo "  - $ds (4 hour timeout)"
done
echo ""
echo "Skipping ${#SKIP_DATASETS[@]} invalid datasets"
echo "=========================================="
echo ""

# Run the optimized batch processor
python "${TAB2IMG_DIR}/run_all_datasets_adaptive.py"

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
    RESULT_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "*_JOB${SLURM_JOB_ID}" | head -1)
    
    echo "✅ SUCCESS!"
    echo ""
    echo "📂 Results saved to:"
    echo "    $RESULT_DIR/"
    echo ""
    echo "📧 Job completion email sent to: aminhajjr@gmail.com"
else
    echo "❌ Some datasets failed, but partial results available"
    echo ""
    echo "Check logs:"
    echo "    Output: $JOB_LOGS_DIR/batch_all_${SLURM_JOB_ID}.out"
    echo "    Error:  $JOB_LOGS_DIR/batch_all_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
exit $EXIT_CODE
