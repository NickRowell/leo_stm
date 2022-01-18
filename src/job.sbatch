#!/bin/sh
#
# Can be launched from the shell using $./job.sbatch in which case SLURM will not be used and images will be processed in sequence.
# Can be launched with SLURM using /usr/bin/sbatch job.sbatch for parallel processing.
# sbatch options:
#
#SBATCH --array=0-7
#SBATCH --job=fireopal

/usr/local/anaconda/3.9/bin/python3 batch_job.py
