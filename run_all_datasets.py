#!/usr/bin/env python3
"""
Batch processor for Table2Image across all OpenML datasets
Organized output structure with separate folders for models, CSVs, logs, and LaTeX
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from datetime import datetime

def create_output_structure(base_output_dir, job_id):
    """
    Create organized folder structure for results
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    run_name = f"{timestamp}_JOB{job_id}"
    
    run_dir = os.path.join(base_output_dir, run_name)
    
    # Create subdirectories
    subdirs = {
        'models': os.path.join(run_dir, 'models'),
        'csv': os.path.join(run_dir, 'csv'),
        'latex': os.path.join(run_dir, 'latex'),
        'logs': os.path.join(run_dir, 'logs')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Create README with run information
    readme_path = os.path.join(run_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f"Table2Image Batch Processing Results\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFolder Structure:\n")
        f.write(f"  models/  - Trained PyTorch models (.pt files)\n")
        f.write(f"  csv/     - Results in CSV format\n")
        f.write(f"  latex/   - LaTeX tables for paper\n")
        f.write(f"  logs/    - Processing logs (JSONL format)\n")
    
    return run_dir, subdirs

def find_datasets(datasets_dir):
    """
    Find all dataset files in folder structure
    Each dataset is in its own folder with potentially different filename
    """
    dataset_files = []
    datasets_path = Path(datasets_dir)
    
    # Iterate through each folder
    for folder in sorted(datasets_path.iterdir()):
        if not folder.is_dir():
            continue
        
        # Search for data files in this folder
        found_files = []
        for pattern in ['*.csv', '*.arff', '*.data']:
            found_files.extend(list(folder.glob(pattern)))
        
        if found_files:
            # Use the first file found (assuming one data file per folder)
            dataset_files.append(found_files[0])
            if len(found_files) > 1:
                print(f"[INFO] Folder '{folder.name}' has {len(found_files)} files, using: {found_files[0].name}")
        else:
            print(f"[WARNING] No data files found in folder: {folder.name}")
    
    return dataset_files

def run_single_dataset(dataset_path, subdirs, script_path, timeout):
    """
    Run Table2Image on a single dataset
    """
    # Use folder name as dataset name (not filename)
    dataset_name = dataset_path.parent.name
    output_path = os.path.join(subdirs['models'], dataset_name)
    
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")
    print(f"Folder: {dataset_path.parent.name}")
    print(f"File: {dataset_path.name}")
    print(f"Output: {output_path}.pt")
    
    # Build command
    cmd = [
        'python', script_path,
        '--data', str(dataset_path),
        '--save_dir', output_path
    ]
    
    # Run with timeout
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS in {elapsed:.1f}s")
            return {
                'status': 'success',
                'dataset': dataset_name,
                'elapsed_time': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ FAILED (exit code {result.returncode})")
            stderr_preview = result.stderr[-1000:] if result.stderr else "No error output"
            print(f"Error preview:\n{stderr_preview}")
            return {
                'status': 'failed',
                'dataset': dataset_name,
                'elapsed_time': elapsed,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏱️  TIMEOUT after {elapsed:.1f}s ({timeout}s limit)")
        return {
            'status': 'timeout',
            'dataset': dataset_name,
            'elapsed_time': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 EXCEPTION: {str(e)}")
        return {
            'status': 'exception',
            'dataset': dataset_name,
            'elapsed_time': elapsed,
            'error': str(e)
        }

def parse_results_jsonl(subdirs):
    """
    Parse results.jsonl and create summary DataFrame
    """
    results_file = os.path.join(subdirs['logs'], 'results.jsonl')
    
    if not os.path.exists(results_file):
        print(f"[WARNING] No results file found at {results_file}")
        return None
    
    # Read all results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    
    if not results:
        print("[WARNING] No valid results found in results.jsonl")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by accuracy descending
    df = df.sort_values('best_accuracy', ascending=False)
    
    return df

def create_summary_tables(df, subdirs, run_dir):
    """
    Create CSV and LaTeX summary tables with average accuracy
    """
    if df is None or len(df) == 0:
        print("[WARNING] No results to summarize")
        return
    
    # Select columns for summary
    summary_df = df[[
        'dataset', 'num_samples', 'num_features', 'num_classes',
        'best_accuracy', 'best_auc', 'best_epoch'
    ]].copy()
    
    # Calculate statistics
    avg_accuracy = summary_df['best_accuracy'].mean()
    avg_auc = summary_df['best_auc'].mean()
    std_accuracy = summary_df['best_accuracy'].std()
    std_auc = summary_df['best_auc'].std()
    
    # ========== 1. RESULTS SUMMARY CSV ==========
    csv_summary_path = os.path.join(subdirs['csv'], 'results_summary.csv')
    summary_df.to_csv(csv_summary_path, index=False)
    print(f"\n✅ Results summary: {csv_summary_path}")
    
    # ========== 2. DETAILED RESULTS CSV ==========
    csv_detailed_path = os.path.join(subdirs['csv'], 'results_detailed.csv')
    df.to_csv(csv_detailed_path, index=False)
    print(f"✅ Detailed results: {csv_detailed_path}")
    
    # ========== 3. STATISTICS CSV ==========
    stats_data = {
        'Metric': ['Average Accuracy', 'Std Accuracy', 'Average AUC', 'Std AUC',
                   'Best Accuracy', 'Worst Accuracy', 'Datasets >90%', 'Datasets >95%'],
        'Value': [
            f"{avg_accuracy:.2f}",
            f"{std_accuracy:.2f}",
            f"{avg_auc:.4f}",
            f"{std_auc:.4f}",
            f"{summary_df['best_accuracy'].max():.2f}",
            f"{summary_df['best_accuracy'].min():.2f}",
            len(summary_df[summary_df['best_accuracy'] > 90]),
            len(summary_df[summary_df['best_accuracy'] > 95])
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    csv_stats_path = os.path.join(subdirs['csv'], 'statistics.csv')
    stats_df.to_csv(csv_stats_path, index=False)
    print(f"✅ Statistics: {csv_stats_path}")
    
    # ========== 4. LATEX TABLE (Top 20) ==========
    latex_path = os.path.join(subdirs['latex'], 'results_latex.txt')
    with open(latex_path, 'w') as f:
        f.write("% LaTeX Table for Paper - Table2Image Results\n")
        f.write("% Top 20 datasets + average\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Table2Image-VIF Performance on OpenML-CC18 Benchmark}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lrrrccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Dataset} & \\textbf{N} & \\textbf{Features} & "
                "\\textbf{Classes} & \\textbf{Acc (\\%)} & \\textbf{AUC} & \\textbf{Epoch} \\\\\n")
        f.write("\\hline\n")
        
        # Top 20 datasets
        top_n = min(20, len(summary_df))
        for idx, row in summary_df.head(top_n).iterrows():
            dataset_name = row['dataset'].replace('_', '\\_')
            f.write(f"{dataset_name} & "
                   f"{row['num_samples']:,} & "
                   f"{row['num_features']} & "
                   f"{row['num_classes']} & "
                   f"{row['best_accuracy']:.2f} & "
                   f"{row['best_auc']:.4f} & "
                   f"{row['best_epoch']} \\\\\n")
        
        if len(summary_df) > top_n:
            f.write(f"\\multicolumn{{7}}{{c}}{{\\textit{{... {len(summary_df) - top_n} more datasets}}}} \\\\\n")
        
        f.write("\\hline\n")
        f.write(f"\\textbf{{Average}} & "
               f"- & - & - & "
               f"\\textbf{{{avg_accuracy:.2f}}} $\\pm$ {std_accuracy:.2f} & "
               f"\\textbf{{{avg_auc:.4f}}} $\\pm$ {std_auc:.4f} & "
               f"- \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table: {latex_path}")
    
    # ========== 5. COMPARISON TABLE ==========
    comparison_path = os.path.join(subdirs['latex'], 'comparison_table.txt')
    with open(comparison_path, 'w') as f:
        f.write("% Comparison with Other Methods (Table 1 format from paper)\n")
        f.write("% Fill in baseline results from the paper\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison on OpenML-CC18}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Avg Accuracy} & \\textbf{Avg AUC} \\\\\n")
        f.write("\\hline\n")
        
        # Your results
        f.write(f"\\textbf{{Table2Image-VIF (Ours)}} & "
               f"\\textbf{{{avg_accuracy:.2f}}} & "
               f"\\textbf{{{avg_auc:.4f}}} \\\\\n")
        
        # From paper Table 1 (OpenML-CC18 results)
        f.write("\\hline\n")
        f.write("Table2Image (baseline) & 87.66 & 0.9202 \\\\\n")
        f.write("XGBoost & 86.75 & 0.8758 \\\\\n")
        f.write("LightGBM & 86.12 & 0.9116 \\\\\n")
        f.write("CatBoost & 86.26 & 0.9146 \\\\\n")
        f.write("FT-Transformer & 83.13 & 0.9016 \\\\\n")
        f.write("TuneTables & 86.50 & 0.9145 \\\\\n")
        f.write("TabM & 84.11 & 0.8960 \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Comparison table: {comparison_path}")
    
    # ========== 6. PRINT SUMMARY ==========
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total datasets: {len(summary_df)}")
    print(f"")
    print(f"📊 AVERAGE ACCURACY: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"📊 AVERAGE AUC:      {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"")
    print(f"🏆 Best:  {summary_df.iloc[0]['dataset']:30s} {summary_df.iloc[0]['best_accuracy']:.2f}%")
    print(f"📉 Worst: {summary_df.iloc[-1]['dataset']:30s} {summary_df.iloc[-1]['best_accuracy']:.2f}%")
    print(f"")
    print(f"Datasets with >90% accuracy: {len(summary_df[summary_df['best_accuracy'] > 90])}")
    print(f"Datasets with >95% accuracy: {len(summary_df[summary_df['best_accuracy'] > 95])}")
    print(f"{'='*70}\n")
    
    # Update README with final results
    readme_path = os.path.join(run_dir, 'README.txt')
    with open(readme_path, 'a') as f:
        f.write(f"\n\nFinal Results:\n")
        f.write(f"="*50 + "\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets processed: {len(summary_df)}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%\n")
        f.write(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Batch process all OpenML datasets with Table2Image'
    )
    parser.add_argument('--datasets_dir', type=str, required=True,
                       help='Directory containing dataset folders')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base output directory (results/ folder)')
    parser.add_argument('--job_id', type=str, required=True,
                       help='SLURM job ID for folder naming')
    parser.add_argument('--script_path', type=str, required=True,
                       help='Path to run_vif.py')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='Timeout per dataset in seconds')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip datasets that already have results')
    
    args = parser.parse_args()
    
    # Create organized output structure
    print(f"{'='*70}")
    print(f"SETTING UP OUTPUT STRUCTURE")
    print(f"{'='*70}")
    run_dir, subdirs = create_output_structure(args.output_base, args.job_id)
    print(f"Output directory: {run_dir}")
    print(f"  📁 models/  → {subdirs['models']}")
    print(f"  📊 csv/     → {subdirs['csv']}")
    print(f"  📄 latex/   → {subdirs['latex']}")
    print(f"  📝 logs/    → {subdirs['logs']}")
    print(f"{'='*70}\n")
    
    # Find all datasets
    print(f"{'='*70}")
    print(f"DISCOVERING DATASETS")
    print(f"{'='*70}")
    print(f"Searching in: {args.datasets_dir}")
    print(f"")
    
    dataset_files = find_datasets(args.datasets_dir)
    
    print(f"\nFound {len(dataset_files)} datasets:")
    for i, dataset_path in enumerate(dataset_files[:15], 1):
        print(f"  {i:2d}. {dataset_path.parent.name:40s} → {dataset_path.name}")
    if len(dataset_files) > 15:
        print(f"  ... and {len(dataset_files) - 15} more datasets")
    print(f"{'='*70}\n")
    
    if len(dataset_files) == 0:
        print("❌ No datasets found!")
        return 1
    
    # Process each dataset
    results_log = []
    success_count = 0
    failed_count = 0
    timeout_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    for i, dataset_path in enumerate(dataset_files, 1):
        elapsed_hours = (time.time() - start_time) / 3600
        remaining = len(dataset_files) - i
        
        print(f"\n{'='*70}")
        print(f"Progress: {i}/{len(dataset_files)} datasets")
        print(f"Elapsed: {elapsed_hours:.1f}h | Remaining: ~{remaining * 0.5:.1f}h")
        print(f"✅ {success_count} | ❌ {failed_count} | ⏱️ {timeout_count}")
        print(f"{'='*70}")
        
        # Check if already processed
        if args.skip_existing:
            dataset_name = dataset_path.parent.name
            model_path = os.path.join(subdirs['models'], f'{dataset_name}.pt')
            if os.path.exists(model_path):
                print(f"⏭️  SKIPPED: {dataset_name} (model exists)")
                skipped_count += 1
                continue
        
        # Run dataset
        result = run_single_dataset(
            dataset_path=dataset_path,
            subdirs=subdirs,
            script_path=args.script_path,
            timeout=args.timeout
        )
        
        results_log.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        elif result['status'] == 'timeout':
            timeout_count += 1
        else:
            failed_count += 1
        
        # Save progress log
        progress_log_path = os.path.join(subdirs['logs'], 'progress_log.jsonl')
        with open(progress_log_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  ⏱️  Timeout: {timeout_count}")
    print(f"  ⏭️  Skipped: {skipped_count}")
    print(f"{'='*70}\n")
    
    # Create summary tables
    if success_count > 0:
        print("Creating summary tables...")
        df = parse_results_jsonl(subdirs)
        create_summary_tables(df, subdirs, run_dir)
    else:
        print("⚠️  No successful results to summarize")
    
    print(f"\n📂 All results saved to: {run_dir}")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
