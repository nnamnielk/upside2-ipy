#!/bin/bash
#SBATCH --job-name=pab1_native_contacts
#SBATCH --account=pi-trsosnic
#SBATCH --output=native_contacts_%j.out
#SBATCH --error=native_contacts_%j.err  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:30:00
#SBATCH --partition=bigmem

# Check if trajectory file argument was provided
if [ -z "$1" ]; then
    echo "Error: No trajectory file provided"
    echo "Usage: sbatch submit_job.slurm <trajectory_file>"
    exit 1
fi

TRAJECTORY_FILE="$1"
TRAJ_BASENAME=$(basename "$TRAJECTORY_FILE" .up)
OUTPUT_PREFIX="results/${TRAJ_BASENAME}"

# Create results directory if it doesn't exist
mkdir -p results

# Print job information
echo "Starting native contacts analysis for: $TRAJECTORY_FILE"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

# Load required modules (adjust as needed for your cluster)
# module load python/3.8
# module load conda
# conda activate your_environment

# Change to the directory containing the script
cd $SLURM_SUBMIT_DIR

# Run the analysis
python3 native_contacts.py "$TRAJECTORY_FILE" "$OUTPUT_PREFIX" --stride 10

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully at: $(date)"
else
    echo "Analysis failed at: $(date)"
    exit 1
fi
