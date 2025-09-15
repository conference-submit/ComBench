#!/usr/bin/env python3
"""
Script for running PFL comparison with ground truth
Automatically processes all model files
"""

import os
import sys
import json
import time
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compare_with_ground_truth import GroundTruthComparator

def calculate_model_stats(model_name, results, args, total_pfl_results=0, coverage=0, processing_mode='existing_file', processing_time=0):
    """Calculate model statistics, only keeping EM-related statistics"""
    # Statistics
    contained_count_em = 0
    avg_containment_score = 0.0
    
    if results:
        contained_count_em = sum(1 for r in results if r['em_analysis']['patches_contained_in_ground_truth'])
        avg_containment_score = sum(r['em_analysis']['containment_score'] for r in results) / len(results)
    
    return {
        'CS': total_pfl_results,
        'EM': contained_count_em,
        'matched_results': len(results),
        'coverage': coverage,
        'contained_count_em': contained_count_em,
        'avg_containment_score': avg_containment_score,
        'processing_mode': processing_mode,
        'max_workers': args.max_workers,
        'processing_time_seconds': processing_time,
        'avg_time_per_result': processing_time / len(results) if len(results) > 0 else 0
    }

def print_model_stats(model_name, stats, results):
    """Print model statistics, only showing EM-related statistics"""
    matched_results = len(results)
    if matched_results > 0:
        print(f"  Patches contained in ground truth (EM): {stats['EM']}/{matched_results} ({stats['EM']/matched_results*100:.1f}%)")
        print(f"  Average containment score (EM): {stats['avg_containment_score']:.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare PFL results with ground truth')
    parser.add_argument('subfolder', help='PFL results subfolder name (e.g., results-bitcoin)')
    parser.add_argument('--target-model', help='Specific target model name to process (optional, if not provided will process all models)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: min(CPU cores, 8))')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (use serial mode)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Extract repo name and project name from subfolder
    # Format: results-{repo}-{date}
    subfolder_parts = args.subfolder.split('-')
    if len(subfolder_parts) >= 2:
        repo_name = subfolder_parts[1]  # Extract repo name
        project_name = subfolder_parts[1]  # Project name is usually the same as repo name
    else:
        repo_name = "bitcoin"  # Default value
        project_name = "bitcoin"  # Default value
    
    # Set paths - script is now in Agent directory, need to go up 4 levels to project root
    base_dir = Path(__file__).parent.parent.parent.parent
    
    # PFL results directory
    pfl_results_dir = base_dir / "experiment/agent/Agent" / args.subfolder / project_name
    
    # Ground truth file - dynamically generated based on repo name
    ground_truth_file = base_dir / "find_repair_patch" / f"{repo_name}_repair_analysis.jsonl"
    
    # Repository path - dynamically generated based on repo name, using correct path format
    repo_path = base_dir / "experiment/repos" / f"{repo_name}_repo"
    
    # Output directory - now in Agent directory
    output_dir = base_dir / "experiment/agent/Agent" / args.subfolder / "sem_eq"
    
    # Create output directory (if it doesn't exist)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        print(f"Using existing directory: {output_dir}")
    else:
        print(f"Creating new directory: {output_dir}")
    
    # Check if basic files exist
    if not ground_truth_file.exists():
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return
    
    if not repo_path.exists():
        print(f"Error: {repo_name} repository not found: {repo_path}")
        return
    
    # Find all detailed_results.jsonl files
    if args.target_model:
        # Process specified model
        model_files = [f"{args.target_model}_detailed_results.jsonl"]
    else:
        # Find all model files
        model_files = []
        if pfl_results_dir.exists():
            for file in pfl_results_dir.glob("*_detailed_results.jsonl"):
                model_files.append(file.name)
    
    if not model_files:
        print(f"Error: No detailed_results.jsonl files found in {pfl_results_dir}")
        return
    
    print(f"=== Configuration Information ===")
    print(f"Repo name: {repo_name}")
    print(f"Project name: {project_name}")
    print(f"Ground truth file: {ground_truth_file}")
    print(f"Repo path: {repo_path}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel processing: {'Enabled' if not args.no_parallel else 'Disabled'}")
    if not args.no_parallel:
        import multiprocessing as mp
        max_workers = args.max_workers or min(mp.cpu_count(), 8)
        print(f"Maximum worker processes: {max_workers}")
    print(f"Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"Found {len(model_files)} model files to process:")
    for model_file in model_files:
        print(f"  - {model_file}")
    print()
    
    # Process each model file
    total_results = {}
    total_start_time = time.time()
    skipped_models = []
    processed_models = []
    
    for model_file in model_files:
        model_name = model_file.replace("_detailed_results.jsonl", "")
        print(f"\n=== Processing Model: {model_name} ===")
        
        model_start_time = time.time()
        
        pfl_results_file = pfl_results_dir / model_file
        output_file = output_dir / f"{model_name}_comparison.jsonl"
        
        if not pfl_results_file.exists():
            print(f"Warning: PFL results file not found: {pfl_results_file}")
            continue
        
        # Check if output file already exists, if so skip processing but read results
        if output_file.exists():
            print(f"Output file already exists, reading existing results for model {model_name}: {output_file}")
            skipped_models.append(model_name)
            # Read existing results file and calculate statistics
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = [json.loads(line.strip()) for line in f if line.strip()]
                
                # Calculate statistics
                # For existing files, CS should equal the number of entries in the results file
                model_stats = calculate_model_stats(model_name, results, args, total_pfl_results=len(results))
                total_results[model_name] = model_stats
                processed_models.append(model_name)
                
                print(f"Read {len(results)} existing results")
                print_model_stats(model_name, model_stats, results)
                
            except Exception as e:
                print(f"Error reading existing results file: {e}")
            continue
        
        print(f"PFL results file: {pfl_results_file}")
        print(f"Output file: {output_file}")
        
        # Create comparator and run analysis
        comparator = GroundTruthComparator(
            pfl_results_file=str(pfl_results_file),
            ground_truth_file=str(ground_truth_file),
            repo_path=str(repo_path),
            output_file=str(output_file),
            max_workers=args.max_workers,
            use_parallel=not args.no_parallel
        )
        
        # Set verbose logging level
        if args.verbose:
            import logging
            comparator.logger.setLevel(logging.DEBUG)
            for handler in comparator.logger.handlers:
                handler.setLevel(logging.DEBUG)
        
        print(f"Starting to call comparator.run_comparison()...")
        try:
            results = comparator.run_comparison()
            print(f"comparator.run_comparison() completed, returned {len(results) if results else 0} results")
        except Exception as e:
            print(f"comparator.run_comparison() encountered exception: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        model_end_time = time.time()
        model_processing_time = model_end_time - model_start_time
        
        # Calculate statistics
        total_pfl_results = len(comparator.pfl_results)
        matched_results = len(results)
        coverage = matched_results/total_pfl_results*100 if total_pfl_results > 0 else 0
        
        # Use helper function to calculate statistics
        model_stats = calculate_model_stats(
            model_name, results, args, 
            total_pfl_results=total_pfl_results,
            coverage=coverage,
            processing_mode='parallel' if not args.no_parallel else 'serial',
            processing_time=model_processing_time
        )
        
        # Save statistics
        total_results[model_name] = model_stats
        processed_models.append(model_name)
        
        # Print statistics
        print(f"\nModel {model_name} analysis completed:")
        print(f"  Processing time: {model_processing_time:.2f} seconds")
        print(f"  Average time per result: {model_processing_time/matched_results:.2f} seconds" if matched_results > 0 else "  Average time per result: N/A")
        print(f"  Number of successfully compiled PFL results: {total_pfl_results}")
        print(f"  Number of matched ground truth: {matched_results}")
        print(f"  Coverage: {coverage:.1f}%")
        
        print_model_stats(model_name, model_stats, results)
    
    # Calculate total processing time
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    # Print overall statistics
    print(f"\n=== Overall Analysis Statistics ===")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Total number of models: {len(total_results)}")
    print(f"Newly processed models: {len(processed_models)}")
    if skipped_models:
        print(f"Models with existing results read: {len(skipped_models)}")
        print(f"Models with existing results: {', '.join(skipped_models)}")
    
    for model_name, stats in total_results.items():
        print(f"{model_name}:")
        print(f"  Processing mode: {stats['processing_mode']}")
        if stats['processing_mode'] == 'parallel' and stats['max_workers']:
            print(f"  Number of worker processes: {stats['max_workers']}")
        print(f"  Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"  Average time per result: {stats['avg_time_per_result']:.2f} seconds")
        print(f"  PFL results: {stats['matched_results']}/{stats['CS']} ({stats['coverage']:.1f}%)")
        if stats['matched_results'] > 0:
            print(f"  Contained in ground truth (EM): {stats['EM']}/{stats['matched_results']} ({stats['EM']/stats['matched_results']*100:.1f}%)")
            print(f"  Average containment score (EM): {stats['avg_containment_score']:.3f}")
    
    # Save overall statistics to file
    summary_file = output_dir / "analysis_summary.json"
    summary_data = {
        'subfolder': args.subfolder,
        'repo_name': repo_name,
        'project_name': project_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_processing_time_seconds': total_processing_time,
        'configuration': {
            'parallel_processing': not args.no_parallel,
            'max_workers': args.max_workers,
            'verbose_logging': args.verbose
        },
        'models': total_results,
        'processed_models': processed_models,
        'skipped_models': skipped_models
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nOverall statistics saved to: {summary_file}")

if __name__ == "__main__":
    main()
