import os, sys, time
from fireopal_settings import *
from fireopal_image_processing import process_list
from fireopal_streak_processing import streak_processing

# is this running as part of a parallel process
if 'SLURM_ARRAY_TASK_COUNT' in os.environ:
    task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    task_min   = int(os.environ['SLURM_ARRAY_TASK_MIN'])
    task_id    = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:
    task_count = 1
    task_min   = 0
    task_id    = 0

# list of files to be processed
filelist = os.listdir(datadirectory)
if len(filelist) == 0:
    print('No files found %s', datadirectory)

# get the date at the beginning of the night
filelist = sorted(filelist)
first = filelist[0]
tok = first.split('_')
try:
    date = tok[1]
except:
    print('cannot find date in filename %s' % first)
    sys.exit()

# set up directory for output products
output           = output + '/' + date + '/'

# will be exception 'FileExistsError' if already exists
if task_id == task_min:
    os.mkdir(output)
    os.mkdir(output + 'detected_streaks')
    os.mkdir(output + 'wcs')
else:
    # wait a bit, make sure the directory has been created
    time.sleep(10)

# list of all the files, split into "my" files
myfilelist = filelist[task_id-task_min : : task_count]
print('processor %d of %d has %d files' % (task_id-task_min, task_count, len(myfilelist)))

# and off we go
process_list(myfilelist, output)

# Launch streak processing from first task
if task_id == task_min:
    # Wait ten mins to allow other tasks in the array to finish
    # TODO Do this better by using slurm --wait.
    time.sleep(600)
    streak_processing(output)
