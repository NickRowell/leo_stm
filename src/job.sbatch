#!/bin/sh
#
# Can be launched from the shell using $./job.sbatch in which case SLURM will not be used and images will be processed in sequence.
# Can be launched with SLURM using $/usr/bin/sbatch job.sbatch for parallel processing asynchronously,
# or $/usr/bin/sbatch --wait job.sbatch for synchronous use in scripts
# sbatch options:
#
#SBATCH --array=0-15
#SBATCH --mem-per-cpu=4G
#SBATCH --job=fireopal
#SBATCH --output=R-%x.%A.%a.out
#SBATCH --error=R-%x.%A.%a.err

# See https://cuillin.roe.ac.uk/projects/documentation/wiki/Proxy_connections_from_workers_and_FCFS_nodes
export http_proxy=http://cuillin:3128/
export https_proxy=http://cuillin:3128/

# Log the execution environment
if [ -z "$SLURM_ARRAY_TASK_COUNT" ]; then
    echo "job.sbatch: running as a standalone process"
else
    echo "job.sbatch: running under SLURM workload management"
fi

echo "Launching python..."

/usr/local/anaconda/3.9/bin/python3 batch_job.py
