# PAB1 Native Contacts Analysis

This directory contains scripts for analyzing native hydrogen bond contacts in PAB1 trajectory files using Slurm job submission.

## Files

- `native_contacts.py` - Main analysis script (converted from notebook)
- `process_batch.py` - Batch processor for multiple trajectories
- `submit_batch.slurm` - Slurm job template for batch processing
- `submit_all_batches.sh` - Master script for intelligent batch submission (recommended)
- `submit_job.slurm` - Slurm job template for individual trajectories  
- `submit_all_pab1_jobs.sh` - Legacy script (has job limit issues)
- `README.md` - This documentation file

## Usage

### Batch Processing (Recommended)

The new batch system processes multiple trajectories sequentially within each job to avoid hitting job submission limits:

```bash
cd pab1/
./submit_all_batches.sh
```

**Features:**
- Automatically detects existing results and skips completed trajectories
- Processes 8 trajectories per batch job (configurable)
- Maintains max 8 concurrent jobs to stay within limits
- Intelligently waits for jobs to complete before submitting new batches
- Handles all 48 trajectories efficiently

### Individual Processing

For single trajectories or testing:

```bash
sbatch submit_job.slurm /path/to/trajectory.up
```

### Direct Analysis (no Slurm)

```bash
python3 native_contacts.py trajectory.up output_prefix --stride 10
```

### Batch Processing (manual)

Process specific trajectories in one job:

```bash
sbatch submit_batch.slurm /path/to/traj1.up /path/to/traj2.up /path/to/traj3.up
```

## Output Files

For each trajectory file `example.up`, the following files will be created in `results/`:

- `example_hbond_energy_maps.npy` - Time series of hydrogen bond energy maps
- `example_hbond_results.pkl` - Complete results object with donors/acceptors info

## Configuration

### Batch Settings (in submit_all_batches.sh)

```bash
BATCH_SIZE=8   # Trajectories per batch job
MAX_JOBS=8     # Maximum concurrent jobs
```

### Job Parameters

**Batch jobs** (`submit_batch.slurm`):
- Memory: 4GB
- Time limit: 4 hours  
- CPUs: 1
- Partition: bigmem

**Individual jobs** (`submit_job.slurm`):
- Memory: 4GB
- Time limit: 30 minutes
- CPUs: 1
- Partition: bigmem

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View batch job logs:
```bash
ls batch_*.out batch_*.err
```

Check results:
```bash
ls results/
```

Resume processing (automatically skips completed):
```bash
./submit_all_batches.sh
```

## Requirements

- Python 3.11 with numpy, pandas, scipy
- Access to `../ff2_rg/utils/mdtraj_upside.py`
- Slurm workload manager
