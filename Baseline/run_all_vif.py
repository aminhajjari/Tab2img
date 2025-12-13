#!/usr/bin/env python3
# this help run all datas via VIF_base.py
"""
Batch runner for Table2Image baseline experiment
Processes all 80 OpenML datasets from tabularDataset/
"""

import os
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

# Configuration
DATASETS_DIR = "/project/def-arashmoh/shahab33/Msc/tabularDataset"
BASELINE_DIR = "/project/def-arashmoh/shahab33/Msc/Tab2img/Baseline"
RUN_SCRIPT = os.path.join(BASELINE_DIR, "VIF_base.py")
RESULTS_BASE = os.path.join(BASELINE_DIR, "results")
TIMEOUT = 7200  # 2 hours per dataset

def find_dataset_file(dataset_dir):
    """Find the main data file in a dataset directory"""
    dataset_name = os.path.basename(dataset_dir)
    
    # Strategy 1: Look for file with same name as directory
    for ext in ['.csv', '.arff', '.data']:
        exact_match = os.path.join(dataset_dir, f"{dataset_name}{ext}")
        if os.path.exists(exact_match):
            return exact_match
    
    # Strategy 2: Look for any file with these extensions
    for ext in ['.csv', '.arff', '.data']:
        files = list(Path(dataset_dir).glob(f'*{ext}'))
        if files:
            return str(files[0])
    
    return None

def main():
    print("="*70)
    print("TABLE2IMAGE BASELINE - BATCH PROCESSOR")
    print("="*70)
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Run script: {RUN_SCRIPT}")
    print(f"Results directory: {RESULTS_BASE}")
    print(f"Timeout per dataset: {TIMEOUT}s ({TIMEOUT//60} minutes)")
    print("="*70)
    
    # Verify run.py exists
    if not os.path.exists(RUN_SCRIPT):
        print(f"\n❌ ERROR: run.py not found at {RUN_SCRIPT}")
        return 1
    
    # Create results directory
    job_id = os.getenv('SLURM_JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
    run_dir = os.path.join(RESULTS_BASE, f"run_{job_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {run_dir}")
    
    # Find all dataset directories
    if not os.path.exists(DATASETS_DIR):
        print(f"\n❌ ERROR: Datasets directory not found: {DATASETS_DIR}")
        return 1
    
    dataset_dirs = sorted([d for d in Path(DATASETS_DIR).iterdir() if d.is_dir()])
    
    print(f"\nFound {len(dataset_dirs)} dataset directories")
    print("="*70)
    
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for idx, dataset_dir in enumerate(dataset_dirs, 1):
        dataset_name = dataset_dir.name
        print(f"\n[{idx}/{len(dataset_dirs)}] Processing: {dataset_name}")
        print("-"*70)
        
        # Find data file
        data_file = find_dataset_file(dataset_dir)
        
        if not data_file:
            print(f"  ⚠️  No data file found in {dataset_dir}")
            print(f"     Checked for: {dataset_name}.csv, {dataset_name}.arff, {dataset_name}.data")
            skip_count += 1
            results.append({
                'dataset': dataset_name,
                'status': 'skipped',
                'reason': 'No data file found'
            })
            continue
        
        print(f"  📁 Data file: {os.path.basename(data_file)}")
        
        # Create output directory for this dataset
        dataset_output = os.path.join(run_dir, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)
        
        # Run the baseline script
        cmd = [
            'timeout', str(TIMEOUT),
            'python', RUN_SCRIPT,
            '--data', data_file,
            '--save_dir', dataset_output
        ]
        
        print(f"  🚀 Running baseline model...")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=BASELINE_DIR
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                print(f"  ✅ Success! ({elapsed:.1f}s)")
                success_count += 1
                results.append({
                    'dataset': dataset_name,
                    'status': 'success',
                    'output_dir': dataset_output,
                    'elapsed_time': elapsed
                })
            elif result.returncode == 124:  # Timeout
                print(f"  ⏱️  Timeout after {TIMEOUT}s")
                fail_count += 1
                results.append({
                    'dataset': dataset_name,
                    'status': 'timeout',
                    'timeout': TIMEOUT
                })
            else:
                print(f"  ❌ Failed (exit code: {result.returncode})")
                # Show last few lines of error
                error_lines = result.stderr.strip().split('\n')[-5:]
                for line in error_lines:
                    print(f"     {line}")
                fail_count += 1
                results.append({
                    'dataset': dataset_name,
                    'status': 'failed',
                    'exit_code': result.returncode,
                    'error': result.stderr[-1000:]  # Last 1000 chars
                })
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")
            fail_count += 1
            results.append({
                'dataset': dataset_name,
                'status': 'error',
                'error': str(e)
            })
        
        # Print progress
        print(f"  Progress: {success_count} success, {fail_count} failed, {skip_count} skipped")
    
    # Save results summary
    summary_file = os.path.join(run_dir, 'batch_summary.json')
    summary = {
        'job_id': job_id,
        'total_datasets': len(dataset_dirs),
        'success': success_count,
        'failed': fail_count,
        'skipped': skip_count,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create simple CSV summary
    csv_file = os.path.join(run_dir, 'batch_summary.csv')
    with open(csv_file, 'w') as f:
        f.write('Dataset,Status,Details\n')
        for r in results:
            details = r.get('elapsed_time', r.get('reason', r.get('error', '')))
            f.write(f"{r['dataset']},{r['status']},{details}\n")
    
    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total datasets: {len(dataset_dirs)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"⚠️  Skipped: {skip_count}")
    print(f"\n📂 Results saved to: {run_dir}")
    print(f"📄 Summary JSON: {summary_file}")
    print(f"📄 Summary CSV: {csv_file}")
    print("="*70)
    
    # List failed datasets if any
    if fail_count > 0:
        print("\n❌ Failed datasets:")
        for r in results:
            if r['status'] in ['failed', 'timeout', 'error']:
                print(f"   - {r['dataset']} ({r['status']})")
    
    return 0 if fail_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
