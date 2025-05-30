#########################################
#                                       #
#          General parameters           #
#                                       #
#########################################

# Location for all outputs
output = ''

# Path to the python executable
pythonpath = ''

# Image dimensions; can these be worked out programmatically?
image_width = 7380
image_height = 4928

#########################################
#                                       #
#      Image processing parameters      #
#                                       #
#########################################

# Input directory of *.NEF images
datadirectory = ''

# Command line parameters for solve-field application
solve_field_options=''

# Cloudy/clear image detection parameters

# Variation 1: Algorithm using pixel intensities (cloudy_or_clear) #

# Pixel intensity threshold for background noise (disregard pixels below
# this value)
cl_background_thresh = 0.05

# Pixel intensity threshold for stars (counts pixels above this intensity)
cl_lower_thresh=0.1

# Standard deviation of Gaussian filter used to blur the raw image [pixels]
cl_sigma=10

# Defines boundary of rectangular region used to determine cloudy/clear status
row1 = int(2000)
row2 = int(2500)
col1 = int(2000)
col2 = int(2500)


# Satellite streak detection parameters

# Thresholded source extraction: number of sigmas above the background for sources.
# A rather low significance level seems necessary in this application.
source_extraction_sigmas = 0.5

# Size of structuring elements used for morphological opening and closing. These remove
# noise and join up fragmented streaks.
opening_kernel_radius = 1
closing_kernel_radius = 4

# Streak classification: sources with fewer connected pixels than this are discarded
sizethresh = 100

# Streak classification: sources with aspect ratios less than this are not streaks
streak_aspect_ratio_min = 10.0
streak_width_min = 15.0

# Parameters for extraction of streak thumbnail image

# Minimum thumbnail image width/height [pix]. This ensures that short streaks still have a large enough
# image surrounding them that a reasonable astrometric solution can be found.
thumbnail_min_diameter = 1000

# Margin added to the image to ensure the ends of the streak are always at least this far from the image edge [pix].
thumbnail_streak_margin = 100

#########################################
#                                       #
#     Streak processing parameters      #
#                                       #
#########################################

# Comma-separated list of email addresses that results are to be sent to
email_addresses=''

# Angle threshold for identifying matching streaks [degrees]
angle_match_threshold = 5

# Distance threshold for identifying matching streaks [pixels]
distance_match_threshold = 20

# Margin applied to image boundary when assessing predicted satellite visibility. This is a simple way to deal with streaks
# that are cut short by the image boundary and for which the prediction is compromised.
margin = 1000
