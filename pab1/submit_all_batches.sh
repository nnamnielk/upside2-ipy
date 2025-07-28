#!/bin/bash
#
# Master batch submission script for PAB1 native contacts analysis.
# This script batches trajectory files and submits them efficiently
# to stay within job submission limits.
#

# Configuration
TRAJ_DIR="/project/trsosnic/okleinmann/oliver/04.HDX/outputs/REMD/Kmarx_Pab1/original"
BATCH_SIZE=8  # Number of trajectories per batch job
MAX_JOBS=8    # Maximum concurrent jobs to maintain
RESULTS_DIR="results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Function to check if results exist for a trajectory
check_results_exist() {
    local traj_file="$1"
    local basename=$(basename "$traj_file" .up)
    local npy_file="${RESULTS_DIR}/${basename}_hbond_energy_maps.npy"
    local pkl_file="${RESULTS_DIR}/${basename}_hbond_results.pkl"
    
    [[ -f "$npy_file" && -f "$pkl_file" ]]
}

# Function to get current job count for this user
get_job_count() {
    squeue -u "$USER" -h | wc -l
}

# Check if trajectory directory exists
if [ ! -d "$TRAJ_DIR" ]; then
    echo "Error: Trajectory directory not found: $TRAJ_DIR"
    exit 1
fi

# Find all trajectory files
echo "Scanning for trajectory files..."
mapfile -t ALL_TRAJS < <(find "$TRAJ_DIR" -name "*.up" | sort)

if [ ${#ALL_TRAJS[@]} -eq 0 ]; then
    echo "No .up trajectory files found in $TRAJ_DIR"
    exit 1
fi

echo "Found ${#ALL_TRAJS[@]} total trajectory files"

# Filter out trajectories that already have results
PENDING_TRAJS=()
COMPLETED_COUNT=0

echo "Checking for existing results..."
for traj in "${ALL_TRAJS[@]}"; do
    if check_results_exist "$traj"; then
        echo "  ✓ $(basename "$traj") - results exist"
        ((COMPLETED_COUNT++))
    else
        echo "  ○ $(basename "$traj") - needs processing"
        PENDING_TRAJS+=("$traj")
    fi
done

echo ""
echo "Summary:"
echo "  Total trajectories: ${#ALL_TRAJS[@]}"
echo "  Already completed: $COMPLETED_COUNT"
echo "  Need processing: ${#PENDING_TRAJS[@]}"

if [ ${#PENDING_TRAJS[@]} -eq 0 ]; then
    echo "All trajectories have been processed!"
    exit 0
fi

# Calculate number of batches needed
NUM_BATCHES=$(( (${#PENDING_TRAJS[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "  Will create $NUM_BATCHES batch jobs (max $BATCH_SIZE trajectories each)"
echo ""

# Submit batch jobs
SUBMITTED_JOBS=0
for ((batch=0; batch<NUM_BATCHES; batch++)); do
    # Check current job count
    CURRENT_JOBS=$(get_job_count)
    
    # Wait if we're at the job limit
    while [ $CURRENT_JOBS -ge $MAX_JOBS ]; do
        echo "Currently have $CURRENT_JOBS jobs running (limit: $MAX_JOBS)"
        echo "Waiting 30 seconds before checking again..."
        sleep 30
        CURRENT_JOBS=$(get_job_count)
    done
    
    # Prepare batch of trajectories
    start_idx=$((batch * BATCH_SIZE))
    end_idx=$((start_idx + BATCH_SIZE - 1))
    
    if [ $end_idx -ge ${#PENDING_TRAJS[@]} ]; then
        end_idx=$((${#PENDING_TRAJS[@]} - 1))
    fi
    
    BATCH_TRAJS=()
    for ((i=start_idx; i<=end_idx; i++)); do
        BATCH_TRAJS+=("${PENDING_TRAJS[$i]}")
    done
    
    echo "Submitting batch $((batch+1))/$NUM_BATCHES with ${#BATCH_TRAJS[@]} trajectories:"
    for traj in "${BATCH_TRAJS[@]}"; do
        echo "  - $(basename "$traj")"
    done
    
    # Submit the batch job
    JOB_ID=$(sbatch submit_batch.slurm "${BATCH_TRAJS[@]}" | grep -o '[0-9]\+')
    
    if [ -n "$JOB_ID" ]; then
        echo "  → Submitted as job $JOB_ID"
        ((SUBMITTED_JOBS++))
    else
        echo "  → ERROR: Failed to submit batch"
    fi
    
    echo ""
    
    # Small delay between submissions
    sleep 2
done

echo "Batch submission complete!"
echo "  Submitted $SUBMITTED_JOBS batch jobs"
echo "  Processing ${#PENDING_TRAJS[@]} trajectories total"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs: batch_*.out and batch_*.err"
echo "Results will be in: $RESULTS_DIR/"
