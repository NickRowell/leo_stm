#!/bin/sh
#
# SLURM sbatch script for launching satellite identification python application to post-process
# the results from the image processing and streak detection application.
#
# Can be launched from the shell using $./job.sbatch in which case SLURM will not be used and images will be processed in sequence.
# Can be launched with SLURM using $/usr/bin/sbatch job.sbatch for parallel processing asynchronously,
# or $/usr/bin/sbatch --wait job.sbatch for synchronous use in scripts
#
# sbatch options: 1 hour wallclock time limit; 1 GB RAM per node
# use 16 nodes; 4G RAM each; send stdout and stderr to files; 3 hour wallclock time limit
#
#SBATCH --job-name=SatelliteIdentification
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

# See https://cuillin.roe.ac.uk/projects/documentation/wiki/Proxy_connections_from_workers_and_FCFS_nodes
export http_proxy=http://cuillin:3128/
export https_proxy=http://cuillin:3128/

# Log the execution environment
if [ -z "$SLURM_JOB_NAME" ]; then
    echo "job.sbatch: running as a standalone process"
else
    echo "job.sbatch: running under SLURM workload management"
fi

echo "Launching python..."

/usr/local/anaconda/3.9/bin/python3 identification.py $outputSubDir /home/nr/fireopal/output/ /detected_streaks /wcs
