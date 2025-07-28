#!/bin/bash
#
# Master submission script for PAB1 native contacts analysis.
# This script submits a separate Slurm job for each .up trajectory file
# in the specified directory.
#

# Directory containing trajectory files
TRAJ_DIR="/project/trsosnic/okleinmann/oliver/04.HDX/outputs/REMD/Kmarx_Pab1/original"

# Check if trajectory directory exists
if [ ! -d "$TRAJ_DIR" ]; then
    echo "Error: Trajectory directory not found: $TRAJ_DIR"
    exit 1
fi

# Count trajectory files
TRAJ_COUNT=$(find "$TRAJ_DIR" -name "*.up" | wc -l)
echo "Found $TRAJ_COUNT trajectory files in $TRAJ_DIR"

if [ $TRAJ_COUNT -eq 0 ]; then
    echo "No .up trajectory files found in the directory"
    exit 1
fi

# Submit jobs for each trajectory file
JOB_COUNT=0
for traj_file in "$TRAJ_DIR"/*.up; do
    if [ -f "$traj_file" ]; then
        echo "Submitting job for: $(basename "$traj_file")"
        sbatch submit_job.slurm "$traj_file"
        JOB_COUNT=$((JOB_COUNT + 1))
        
        # Add small delay to avoid overwhelming the scheduler
        sleep 0.5
    fi
done

echo ""
echo "Successfully submitted $JOB_COUNT jobs"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check results in the ./pab1/results/ directory"
