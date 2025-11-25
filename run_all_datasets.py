#!/usr/bin/env python3
"""
Batch processor for Table2Image across all OpenML datasets
Automatically discovers and processes datasets in folder structure
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

def run_single_dataset(dataset_path, output_dir, script_path, epochs, batch_size, 
                       dataset_root, timeout):
    """
    Run Table2Image on a single dataset
    """
    # Use folder name as dataset name (not filename)
    dataset_name = dataset_path.parent.name
    output_path = os.path.join(output_dir, 'models', dataset_name)
    
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")
    print(f"Folder: {dataset_path.parent.name}")
    print(f"File: {dataset_path.name}")
    print(f"Output: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
            # Print last part of stderr for debugging
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

def parse_results_jsonl(output_dir):
    """
    Parse results.jsonl and create summary DataFrame
    """
    results_file = os.path.join(output_dir, 'results.jsonl')
    
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

def create_summary_tables(df, output_dir):
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
    
    # Calculate averages
    avg_accuracy = summary_df['best_accuracy'].mean()
    avg_auc = summary_df['best_auc'].mean()
    
    # ========== CSV Summary ==========
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV summary saved to: {csv_path}")
    
    # ========== LaTeX Table (Paper Format) ==========
    latex_path = os.path.join(output_dir, 'results_latex.txt')
    with open(latex_path, 'w') as f:
        f.write("% LaTeX Table for Paper - Table2Image Results\n")
        f.write("% Use this in your IEEE paper\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Table2Image-VIF Performance on OpenML-CC18 Benchmark}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lrrrccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Dataset} & \\textbf{Samples} & \\textbf{Features} & "
                "\\textbf{Classes} & \\textbf{Accuracy} & \\textbf{AUC} & \\textbf{Epoch} \\\\\n")
        f.write("\\hline\n")
        
        # Write top 20 datasets (or all if less than 20)
        top_n = min(20, len(summary_df))
        for idx, row in summary_df.head(top_n).iterrows():
            dataset_name = row['dataset'].replace('_', '\\_')  # Escape underscores
            f.write(f"{dataset_name} & "
                   f"{row['num_samples']:,} & "
                   f"{row['num_features']} & "
                   f"{row['num_classes']} & "
                   f"{row['best_accuracy']:.2f} & "
                   f"{row['best_auc']:.4f} & "
                   f"{row['best_epoch']} \\\\\n")
        
        if len(summary_df) > top_n:
            f.write(f"\\multicolumn{{7}}{{c}}{{... and {len(summary_df) - top_n} more datasets}} \\\\\n")
        
        f.write("\\hline\n")
        f.write(f"\\textbf{{Average}} & "
               f"- & - & - & "
               f"\\textbf{{{avg_accuracy:.2f}}} & "
               f"\\textbf{{{avg_auc:.4f}}} & "
               f"- \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table saved to: {latex_path}")
    
    # ========== Comparison Table (for paper) ==========
    comparison_path = os.path.join(output_dir, 'comparison_table.txt')
    with open(comparison_path, 'w') as f:
        f.write("% Comparison with Other Methods (Table 1 format from paper)\n")
        f.write("% Add your baseline results to compare\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison on OpenML-CC18 Benchmark}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Avg Accuracy} & \\textbf{Avg AUC} \\\\\n")
        f.write("\\hline\n")
        
        # Your results
        f.write(f"\\textbf{{Table2Image-VIF (Ours)}} & "
               f"\\textbf{{{avg_accuracy:.2f}}} & "
               f"\\textbf{{{avg_auc:.4f}}} \\\\\n")
        
        # Placeholder for baselines (user should fill these in)
        f.write("\\hline\n")
        f.write("XGBoost & [TODO] & [TODO] \\\\\n")
        f.write("LightGBM & [TODO] & [TODO] \\\\\n")
        f.write("CatBoost & [TODO] & [TODO] \\\\\n")
        f.write("FT-Transformer & [TODO] & [TODO] \\\\\n")
        f.write("TabPFN & [TODO] & [TODO] \\\\\n")
        f.write("TuneTables & [TODO] & [TODO] \\\\\n")
        f.write("TabM & [TODO] & [TODO] \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Comparison table template saved to: {comparison_path}")
    
    # ========== Print Summary Statistics ==========
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total datasets processed: {len(summary_df)}")
    print(f"")
    print(f"📊 AVERAGE ACCURACY: {avg_accuracy:.2f}%  ← Use this for paper comparison!")
    print(f"📊 AVERAGE AUC:      {avg_auc:.4f}")
    print(f"")
    print(f"Best performing dataset:")
    print(f"  {summary_df.iloc[0]['dataset']}: {summary_df.iloc[0]['best_accuracy']:.2f}%")
    print(f"")
    print(f"Worst performing dataset:")
    print(f"  {summary_df.iloc[-1]['dataset']}: {summary_df.iloc[-1]['best_accuracy']:.2f}%")
    print(f"")
    print(f"Datasets with >90% accuracy: {len(summary_df[summary_df['best_accuracy'] > 90])}")
    print(f"Datasets with >95% accuracy: {len(summary_df[summary_df['best_accuracy'] > 95])}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Batch process all OpenML datasets with Table2Image'
    )
    parser.add_argument('--datasets_dir', type=str, required=True,
                       help='Directory containing dataset folders')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--script_path', type=str, required=True,
                       help='Path to run_vif.py')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Root directory for MNIST/FashionMNIST')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='Timeout per dataset in seconds (default: 2 hours)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip datasets that already have results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        print("❌ No datasets found! Check your datasets_dir path.")
        return 1
    
    # Confirmation
    print(f"Ready to process {len(dataset_files)} datasets")
    print(f"Estimated time: ~{len(dataset_files) * args.timeout / 3600:.1f} hours (if all hit timeout)")
    print(f"Typical time: ~{len(dataset_files) * 0.5:.1f} hours (30 min per dataset)")
    print(f"")
    
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
        print(f"Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count}")
        print(f"{'='*70}")
        
        # Check if already processed
        if args.skip_existing:
            dataset_name = dataset_path.parent.name
            model_path = os.path.join(args.output_dir, 'models', f'{dataset_name}.pt')
            if os.path.exists(model_path):
                print(f"⏭️  SKIPPED: {dataset_name} (model exists)")
                skipped_count += 1
                continue
        
        # Run dataset
        result = run_single_dataset(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            script_path=args.script_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset_root=args.dataset_root,
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
        progress_log_path = os.path.join(args.output_dir, 'progress_log.jsonl')
        with open(progress_log_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Datasets found: {len(dataset_files)}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  ⏱️  Timeout: {timeout_count}")
    print(f"  ⏭️  Skipped: {skipped_count}")
    print(f"{'='*70}\n")
    
    # Create summary tables
    if success_count > 0:
        print("Creating summary tables...")
        df = parse_results_jsonl(args.output_dir)
        create_summary_tables(df, args.output_dir)
    else:
        print("⚠️  No successful results to summarize")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
