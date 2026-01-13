#!/bin/bash
#
# Can be launched from the shell using $./job.sbatch in which case SLURM will not be used and images will be processed in sequence.
# Can be launched with SLURM using $/usr/bin/sbatch job.sbatch for parallel processing asynchronously,
# or $/usr/bin/sbatch --wait job.sbatch for synchronous use in scripts
#
# sbatch options: use 16 nodes; 6G RAM each; send stdout and stderr to files; 3 hour wallclock time limit
#
# 2025/04/15 -> increased mem-per-cpu from 4G to 5G to see if that stops oom_kill events
# 2025/05/20 -> switched from astrometry.net api to solve-field command line application;
#               increased mem-per-cpu from 5G to 8G to compensate for greater memory footprint.
#               Switched from Bourne shell (sh) to bash in order to load module astrometry.net
#
#SBATCH --array=0-15
#SBATCH --mem-per-cpu=8G
#SBATCH --job=fireopal
#SBATCH --output=R-%x.%A.%a.out
#SBATCH --error=R-%x.%A.%a.err
#SBATCH --time=3:00:00

# See https://cuillin.roe.ac.uk/projects/documentation/wiki/Proxy_connections_from_workers_and_FCFS_nodes
export http_proxy=http://cuillin:3128/
export https_proxy=http://cuillin:3128/

# Configure the modules environment (necessary when launching from a cronjob)
. /usr/share/modules/init/bash

# Load astrometry.net module to access command line application
module load astrometry.net

# Log the execution environment
if [ -z "$SLURM_ARRAY_TASK_COUNT" ]; then
    echo "job.sbatch: running as a standalone process"
else
    echo "job.sbatch: running under SLURM workload management"
fi

echo "Launching python..."

/usr/local/anaconda/3.13/bin/python3 batch_job.py
