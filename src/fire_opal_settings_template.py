# Algorithmic and configuration parameters for the FireOPAL scripts.

# Text file produced by fire_opal_postprocess.py
txtpath = ''

# Location for all outputs of fire_opal_v2.py
output = ''

# Input directory of *.NEF images for fire_opal_v2.py
datadirectory = ''

# Path to the python executable
pythonpath = ''

# Location of the Astrometry.NET client
clientpath = '../nova_client.py'

# An API key is needed to access astrometry.net. The API key is linked
# to a specific user account.
apikey = ''


# Cloudy/clear image detection parameters

# Variation 1: Algorithm using pixel intensities (cloudy_or_clear) #

cl_background_thresh = 0.1 
# Pixel intensity threshold for background noise (disregard pixels below
# this value)
cl_lower_thresh=0.15 
# Pixel intensity threshold for stars (counts pixels above this intensity)
cl_sigma=10 
# Sigma used in Gaussian filter
row1 = int(2000) 
row2 = int(2500)
col1 = int(2000)
col2 = int(2500)
# Row and column values for subimage

# Variation 2: Algorithm using histogram bins (cloudy_or_clear_alt) #
# Not being used at the moment

cl_curve_thresh = 0.3
# Fraction of maximum histogram value where curve width is measured
axis_comp = 20.0
# How many times longer should the right-hand tail of the curve be?

# Satellite streak detection parameters
    
# Canny edge dector
definitely_not_an_edge = 90 # Below this gradient value, pixel is not an edge
definitely_an_edge = 180 # Above this gradient value, pixel is definitely an edge

# Hough line transform 
line_votes = 100 # How many votes for something to count as line (e.g. length of line)

# Section of image that contains streak
box_length = 600
# Distance in pixels from centre of box to edge, e.g. 250 would give a 500x500 box

# Trail assembly and orbit determination parameters

floor_scale = 100
# The slope is multiplied by this factor. The floor function used in grouping
# rounds to the nearest integer. Since slopes in test data can have values
# such as 0.4578, 1.872, etc., rounding to whole integers isn't useful. 
# Multiply by floor_scale to get 45.78 and 187.2 - now integer rounding works!
# Effectively you are rounding to the 2nd decimal place. To round to the 3rd
# decimal place, change floor_scale to 1000.
