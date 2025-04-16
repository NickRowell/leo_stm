import os, sys, time
from pathlib import Path
from settings import *
from image_processing import process_images
from streak_processing import process_streaks

print('Starting up...', flush=True)

# is this running as part of a parallel process
if 'SLURM_ARRAY_TASK_COUNT' in os.environ:
    task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    task_min   = int(os.environ['SLURM_ARRAY_TASK_MIN'])
    task_id    = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print('Executing under SLURM workload management', flush=True)
else:
    task_count = 1
    task_min   = 0
    task_id    = 0
    print('Executing as a standalone process', flush=True)

# List of files in the datadirectory
realdatadir = os.path.realpath(datadirectory)
try:
    filelist = os.listdir(datadirectory)
except NotADirectoryError:
    # Workaround for intermittent issue
    print('Encountered NotADirectoryError; realpath of %s is %s' % (datadirectory, realdatadir), flush=True)
    quit()

# Remove files that are not NEF images
filelist = [f for f in filelist if f.endswith('.NEF')]

if len(filelist) == 0:
    print('No NEF files found %s' % datadirectory, flush=True)
    sys.exit()

print('Found %d NEF files in %s' % (len(filelist), datadirectory), flush=True)

# get the date at the beginning of the night
filelist = sorted(filelist)
first = filelist[0]
tok = first.split('_')
try:
    date = tok[1]
except:
    print('Cannot find date in filename %s' % first, flush=True)
    sys.exit()

# set up directory for output products
output = output + '/' + date + '/'

if task_id == task_min:
    # First task creates the output directories
    try:

        print('Creating output directory %s' % output, flush=True)

        os.mkdir(output)
        os.mkdir(output + 'detected_streaks')
        os.mkdir(output + 'streak_images')
        os.mkdir(output + 'wcs')

        # Create symlink to latest processing results, in parent folder of the datadirectory
        datapath = Path(datadirectory)
        datapathparent = datapath.parent.absolute()
        procres = str(datapathparent) + '/processed'
        # Can't overwrite an existing symlink; must create temporary link then rename it
        procres_TMP = str(datapathparent) + '/processed_TMP'
        os.symlink(output, procres_TMP)
        os.rename(procres_TMP, procres)
    except FileExistsError:
        print('Output directories already exist', flush=True)
else:
    # Other tasks pause for 2 mins, to allow time for the directory to be created.
    time.sleep(120)

# list of all the files, split into "my" files
myfilelist = filelist[task_id-task_min : : task_count]
print('Processor %d of %d has %d files' % (task_id-task_min, task_count, len(myfilelist)), flush=True)

# and off we go
process_images(myfilelist, output)

# Launch streak processing from first task
if task_id == task_min:
    # Wait ten mins to allow other tasks in the array to finish
    # TODO Do this better by using slurm --wait.
    time.sleep(600)

    process_streaks(output, date)
