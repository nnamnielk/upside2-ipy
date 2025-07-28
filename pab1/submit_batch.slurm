#!/bin/bash
#SBATCH --job-name=pab1_batch
#SBATCH --account=pi-trsosnic
#SBATCH --output=batch_%j.out
#SBATCH --error=batch_%j.err  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --partition=bigmem

# Check if trajectory files were provided
if [ $# -eq 0 ]; then
    echo "Error: No trajectory files provided"
    echo "Usage: sbatch submit_batch.slurm <trajectory_file1> <trajectory_file2> ..."
    exit 1
fi

# Print job information
echo "Starting batch native contacts analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Processing ${#} trajectory files:"
for file in "$@"; do
    echo "  - $(basename "$file")"
done

# Load required modules (adjust as needed for your cluster)
# module load python/3.8
# module load conda
# conda activate your_environment

# Change to the directory containing the script
cd $SLURM_SUBMIT_DIR

# Create results directory if it doesn't exist
mkdir -p results

# Run the batch analysis
python3 process_batch.py "$@" --stride 10

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Batch analysis completed successfully at: $(date)"
else
    echo "Batch analysis failed at: $(date)"
    exit 1
fi
