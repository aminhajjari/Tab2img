#!/usr/bin/env python3
"""
Batch Processing Script for Table2Image - Matches run_vif.py arguments

This script processes all datasets in tabularDataset/ folder structure:
    tabularDataset/
        adult/phpMawTba.arff
        diabetes/dataset.csv
        ...

Author: Shahab
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
import subprocess
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_datasets(datasets_dir: str) -> list:
    """
    Find all dataset files in subdirectories.
    
    Expected structure:
        datasets_dir/
            adult/
                phpMawTba.arff
            diabetes/
                dataset.csv
            ...
    
    Returns:
        List of tuples: (dataset_name, file_path)
    """
    datasets = []
    valid_extensions = ['.csv', '.arff', '.data']
    
    logger.info(f"Scanning: {datasets_dir}")
    
    for folder_name in sorted(os.listdir(datasets_dir)):
        folder_path = os.path.join(datasets_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Find first data file in folder
        for file_name in os.listdir(folder_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if file_ext in valid_extensions:
                file_path = os.path.join(folder_path, file_name)
                datasets.append((folder_name, file_path))
                logger.info(f"  ✓ {folder_name} -> {file_name}")
                break
    
    logger.info(f"Found {len(datasets)} datasets")
    return datasets


def run_single_dataset(dataset_name: str, dataset_path: str, 
                       output_dir: str, args) -> dict:
    """Run Table2Image on a single dataset."""
    
    logger.info("=" * 80)
    logger.info(f"Processing: {dataset_name}")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Create output directory for this dataset
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Model save path
    model_save_path = os.path.join(dataset_output_dir, 'model')
    
    # Build command matching run_vif.py arguments
    cmd = [
        sys.executable,
        args.script_path,
        '--csv', dataset_path,              # Note: uses --csv not --input_file
        '--save_dir', model_save_path,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--dataset_root', args.dataset_root
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=os.path.dirname(args.script_path)  # Run from script directory
        )
        
        elapsed_time = time.time() - start_time
        
        # Save stdout/stderr
        with open(os.path.join(dataset_output_dir, 'stdout.log'), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(dataset_output_dir, 'stderr.log'), 'w') as f:
            f.write(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS in {elapsed_time:.1f}s")
            
            # Parse results
            accuracy, auc = parse_output(result.stdout)
            
            return {
                'dataset': dataset_name,
                'status': 'success',
                'accuracy': accuracy,
                'auc': auc,
                'time_seconds': elapsed_time,
                'samples': extract_samples(result.stdout),
                'features': extract_features(result.stdout),
                'classes': extract_classes(result.stdout),
                'error': None
            }
        else:
            logger.error(f"❌ FAILED")
            logger.error(f"Error: {result.stderr[:500]}")
            
            return {
                'dataset': dataset_name,
                'status': 'failed',
                'accuracy': None,
                'auc': None,
                'time_seconds': elapsed_time,
                'samples': None,
                'features': None,
                'classes': None,
                'error': result.stderr[:200]
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"⏱️ TIMEOUT after {args.timeout}s")
        
        return {
            'dataset': dataset_name,
            'status': 'timeout',
            'accuracy': None,
            'auc': None,
            'time_seconds': elapsed_time,
            'samples': None,
            'features': None,
            'classes': None,
            'error': f'Timeout ({args.timeout}s)'
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"💥 EXCEPTION: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'dataset': dataset_name,
            'status': 'error',
            'accuracy': None,
            'auc': None,
            'time_seconds': elapsed_time,
            'samples': None,
            'features': None,
            'classes': None,
            'error': str(e)[:200]
        }


def parse_output(output: str) -> tuple:
    """Extract accuracy and AUC from output."""
    accuracy = None
    auc = None
    
    for line in output.split('\n'):
        if 'Best model accuracy:' in line:
            try:
                parts = line.split('accuracy:')[1].split()
                accuracy = float(parts[0])
            except:
                pass
        
        if 'Best AUC:' in line:
            try:
                auc = float(line.split('AUC:')[1].strip())
            except:
                pass
    
    return accuracy, auc


def extract_samples(output: str) -> int:
    """Extract number of samples from output."""
    for line in output.split('\n'):
        if 'Total samples:' in line:
            try:
                return int(line.split(':')[1].strip())
            except:
                pass
    return None


def extract_features(output: str) -> int:
    """Extract number of features from output."""
    for line in output.split('\n'):
        if 'Total features:' in line or 'Number of features:' in line:
            try:
                return int(line.split(':')[1].strip())
            except:
                pass
    return None


def extract_classes(output: str) -> int:
    """Extract number of classes from output."""
    for line in output.split('\n'):
        if 'Number of classes:' in line:
            try:
                return int(line.split(':')[1].strip())
            except:
                pass
    return None


def save_results(results: list, output_dir: str):
    """Save and display results."""
    df = pd.DataFrame(results)
    df = df.sort_values('dataset')
    
    # Save CSV
    csv_file = os.path.join(output_dir, 'results_all_datasets.csv')
    df.to_csv(csv_file, index=False)
    logger.info(f"Results saved: {csv_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = df[df['status'] == 'success']
    failed = df[df['status'] != 'success']
    
    print(f"Total: {len(df)}")
    print(f"Success: {len(successful)} ({100*len(successful)/len(df):.1f}%)")
    print(f"Failed: {len(failed)}")
    
    if len(successful) > 0:
        print(f"\nPerformance:")
        print(f"  Avg Accuracy: {successful['accuracy'].mean():.4f} ± {successful['accuracy'].std():.4f}")
        print(f"  Avg AUC: {successful['auc'].mean():.4f} ± {successful['auc'].std():.4f}")
        print(f"  Avg Time: {successful['time_seconds'].mean()/60:.1f} minutes")
        
        print("\n" + "=" * 80)
        print("TOP 10 DATASETS")
        print("=" * 80)
        top10 = successful.nlargest(10, 'accuracy')[['dataset', 'accuracy', 'auc', 'time_seconds']]
        for _, row in top10.iterrows():
            print(f"  {row['dataset']:30s} | Acc: {row['accuracy']:.4f} | AUC: {row['auc']:.4f} | Time: {row['time_seconds']/60:.1f}m")
    
    if len(failed) > 0:
        print("\n" + "=" * 80)
        print("FAILED DATASETS")
        print("=" * 80)
        for _, row in failed.iterrows():
            print(f"  {row['dataset']:30s} | Status: {row['status']:10s} | Error: {str(row['error'])[:50]}")
    
    # Save LaTeX table
    latex_file = os.path.join(output_dir, 'results_latex.txt')
    with open(latex_file, 'w') as f:
        f.write("% LaTeX table for paper\n\n")
        if len(successful) > 0:
            f.write(successful[['dataset', 'samples', 'features', 'classes', 'accuracy', 'auc']].to_latex(
                index=False,
                float_format="%.4f",
                caption="Table2Image results on benchmark datasets",
                label="tab:results"
            ))
    logger.info(f"LaTeX table saved: {latex_file}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batch process datasets with Table2Image")
    
    parser.add_argument('--datasets_dir', required=True, help='Directory with dataset folders')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--script_path', required=True, help='Path to run_vif.py')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset_root', type=str, default='/tmp/datasets', help='MNIST/FashionMNIST root')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout per dataset (seconds)')
    parser.add_argument('--start_from', type=str, help='Resume from this dataset')
    parser.add_argument('--max_datasets', type=int, help='Limit number of datasets')
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.datasets_dir):
        logger.error(f"Datasets dir not found: {args.datasets_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.script_path):
        logger.error(f"Script not found: {args.script_path}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING START")
    logger.info("=" * 80)
    logger.info(f"Datasets: {args.datasets_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Script: {args.script_path}")
    logger.info(f"Config: {args.epochs} epochs, batch {args.batch_size}, timeout {args.timeout}s")
    
    # Find datasets
    datasets = find_datasets(args.datasets_dir)
    
    if len(datasets) == 0:
        logger.error("No datasets found!")
        sys.exit(1)
    
    # Apply filters
    if args.start_from:
        datasets = [(n, p) for n, p in datasets if n >= args.start_from]
        logger.info(f"Starting from: {args.start_from} ({len(datasets)} remaining)")
    
    if args.max_datasets:
        datasets = datasets[:args.max_datasets]
        logger.info(f"Limited to {args.max_datasets} datasets")
    
    # Process
    results = []
    total_start = time.time()
    
    for i, (name, path) in enumerate(datasets, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{len(datasets)}] {name}")
        logger.info(f"{'='*80}")
        
        result = run_single_dataset(name, path, args.output_dir, args)
        results.append(result)
        
        # Save intermediate results
        save_results(results, args.output_dir)
    
    # Final
    total_time = time.time() - total_start
    logger.info(f"\n{'='*80}")
    logger.info("BATCH COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Avg per dataset: {total_time/len(datasets)/60:.1f} minutes")
    
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
