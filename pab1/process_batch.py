#!/usr/bin/env python3
"""
Batch processor for native contacts analysis.
Processes multiple trajectory files sequentially in one job.
"""

import os
import sys
import argparse
from pathlib import Path

# Import the analysis function
from native_contacts import run_hbond_analysis, save_results

def check_results_exist(output_prefix):
    """Check if results already exist for this trajectory."""
    npy_file = f"{output_prefix}_hbond_energy_maps.npy"
    pkl_file = f"{output_prefix}_hbond_results.pkl"
    return os.path.exists(npy_file) and os.path.exists(pkl_file)

def process_trajectory_batch(trajectory_files, stride=10):
    """Process a batch of trajectory files sequentially."""
    processed = 0
    skipped = 0
    failed = 0
    
    print(f"Starting batch processing of {len(trajectory_files)} trajectories")
    print(f"Using stride: {stride}")
    print("=" * 60)
    
    for i, traj_file in enumerate(trajectory_files, 1):
        traj_path = Path(traj_file)
        traj_basename = traj_path.stem
        output_prefix = f"results/{traj_basename}"
        
        print(f"\n[{i}/{len(trajectory_files)}] Processing: {traj_basename}")
        
        # Check if results already exist
        if check_results_exist(output_prefix):
            print(f"  → Results already exist, skipping")
            skipped += 1
            continue
            
        # Check if trajectory file exists
        if not os.path.exists(traj_file):
            print(f"  → ERROR: Trajectory file not found")
            failed += 1
            continue
            
        try:
            # Run the analysis
            print(f"  → Running analysis...")
            results = run_hbond_analysis(traj_file, stride=stride)
            
            # Save results
            print(f"  → Saving results...")
            save_results(results, output_prefix)
            
            processed += 1
            print(f"  → SUCCESS: Completed {traj_basename}")
            
        except Exception as e:
            print(f"  → ERROR: Failed to process {traj_basename}: {str(e)}")
            failed += 1
            continue
    
    print("\n" + "=" * 60)
    print(f"Batch processing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(trajectory_files)}")
    
    return processed, skipped, failed

def main():
    parser = argparse.ArgumentParser(description='Process a batch of trajectory files')
    parser.add_argument('trajectory_files', nargs='+', help='List of trajectory files to process')
    parser.add_argument('--stride', type=int, default=10, help='Stride for trajectory reading (default: 10)')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Process the batch
    processed, skipped, failed = process_trajectory_batch(args.trajectory_files, args.stride)
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
