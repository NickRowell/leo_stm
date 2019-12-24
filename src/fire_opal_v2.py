# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:56:56 2019

@author: Maureen

This script runs the Fire Opal program.

Structure: 
    - Defines functions
    - Runs FOR loop that batch processes image files from a directory and
    extracts data
    
Requirements:
    Astrometry.net API client
    fire_opal_settings.py
    Rawpy, cv2, astropy, datetime, os, scipy, numpy


"""
from fire_opal_settings import *
import os, rawpy, cv2, datetime, nova_client
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS
from astropy.io import fits

# TODO Move these to settings file
image_width = 7380
image_height = 4928

def convert_to_grey(rgbimage):
    
    """ This function converts an RGB image to a 
    greyscale image by averaging the RGB values.
    
    Input: RGB image
    Output: Greyscale image """

    r = rgbimage[:,:,0]
    g = rgbimage[:,:,1]
    b = rgbimage[:,:,2]

    # Apply 3x3 moving average kernel to beat down shot noise
    kernel = np.ones((3,3),np.float32)/9
    r = cv2.filter2D(r,-1,kernel)
    g = cv2.filter2D(g,-1,kernel)
    b = cv2.filter2D(b,-1,kernel)

    # Measure backgrounds of each image
    r_bkg = cv2.medianBlur(r, 71)
    g_bkg = cv2.medianBlur(g, 71)
    b_bkg = cv2.medianBlur(b, 71)

    # Compute estimate of standard deviation in each pixel
    var = 1.0 / (1.0/r_bkg + 1.0/g_bkg + 1.0/b_bkg)

    # Compute variance weighted average of background-subtracted image
    signal = ((r.astype('float64')-r_bkg)/r_bkg + (g.astype('float64')-g_bkg)/g_bkg + (b.astype('float64')-b_bkg)/b_bkg) * var

    # Simple greyscale image for use with astrometry.NET and visualisation
    grey = np.add(r * 0.1, g * 0.6, b * 0.3)
    
    #cv2.imwrite(str(output) + '/detected_streaks/' + 'grey_new.png', grey)

    return signal, var, grey

def cloudy_or_clear(greyimage):
    
    """ 
    This function sorts greyscale images of the night sky into two
    categories: clear or cloudy. Returns Boolean True or False.
    
    Inputs: Greyscale image, upper intensity bound of background, lower
    intensity bound for stars, Gaussian filter sigma.
    Output: True if clear, False if cloudy 
    
    """

    # Make a defocused copy of original image and subtract from original
    # A cloudy input image results in pure noise, while a clear input image
    # has points.
    c = np.abs(greyimage.astype('float64') - gaussian_filter(greyimage, cl_sigma).astype('float64'))

    # Extract rectangular region for cloudy/clear determination
    subimage = c[row1:row2, col1:col2]

    # Count the number of pixels in the subimage below an intensity threshold 
    # defined as the background.
    ignore = len(subimage[np.where(subimage<=cl_background_thresh)])

    # Calculate the percentage of pixels above the brightness threshold, after
    # low-intensity background pixels have been disregarded.
    # TODO: this 500*500 should be replaced with the image dimensions from the settings
    source = len(subimage[np.where(subimage>cl_lower_thresh)])
    perc = 100*(source)/(500*500 - ignore + 1)

    if perc > 0.:
        return True
    else:
        return False

def circular_kernel(radius):

    """
    Construct a circular kernel of the given radius.
    """

    width = 2*radius + 1
    kernel = np.zeros((width, width), np.uint8)
    for i in range(0, width):
        for j in range(0, width):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius**2:
                kernel[i][j] = int(1)
    return kernel

def detect_streak(pixels):

    """
    This function computes morphological properties of the source defined by the given set of pixels,
    to enable streak classification.

    Inputs: array of (x,y) pixel coordinates
    Outputs: ratio of the length to width of the source, and the (x,y) coordinates of the points
             at each end of the source along it's longest axis, allowing a margin for the PSF size.
    
    """

    # pixels is a 'size' x 2 array containing the pixel coordinates of the source pixels.
    # Compute Hesse normal line fitting these points.
    mx = 0
    my = 0
    mx2 = 0
    my2 = 0
    mxy = 0

    for x,y in pixels:
        mx += x
        my += y
        mx2 += x*x
        my2 += y*y
        mxy += x*y

    mx /= len(pixels)
    my /= len(pixels)
    mx2 /= len(pixels)
    my2 /= len(pixels)
    mxy /= len(pixels)

    # Orientation parameter
    theta = 0.5 * np.arctan2(2 * (mx*my - mxy), (mx*mx - mx2)-(my*my - my2))

    # Unit vector normal to the line
    n = np.array([np.cos(theta), np.sin(theta)])

    # Unit vector parallel to the line
    t = np.array([-np.sin(theta), np.cos(theta)])

    # Distance of closest approach to the origin
    r = np.dot(n, [mx, my])

    # Points (x,y) on the line satisfy nx*x + ny*y - r = 0

    # Compute width of source perpendicular to the line
    d_max = 0
    d_min = 0
    # Compute length of source along the line
    # Initialise length range using position of the mean point (mx,my)
    e_max = np.dot(t, [mx, my])
    e_min = e_max

    for xy in pixels:

        # Perpendicular distance from line to xy
        d = np.dot(n, xy) - r
        # Record min/max value
        d_max = np.maximum(d_max, d)
        d_min = np.minimum(d_min, d)

        # Parallel distance from xy to location on line closest to the origin
        e = np.dot(t, xy)
        # Record min/max value
        e_max = np.maximum(e_max, e)
        e_min = np.minimum(e_min, e)

    # Get the width & length
    width = d_max - d_min
    length = e_max - e_min

    # Get the end points, subtracting half the width from
    # each end point to correct for PSF size.
    a = r * n + (e_min + width/2) * t
    b = r * n + (e_max - width/2) * t

    return length, width, a, b

def process_image(datadirectory, file, streaks_file, processed_images, output):

    # Read raw NEF image
    raw = rawpy.imread(datadirectory + file)

    # Convert to standard RGB pixels [0:255]
    rgb = raw.postprocess()

    # Use green channel to estimate cloudiness
    is_it_clear = cloudy_or_clear(rgb[:,:,1])
    is_it_clear = True

    # Skip cloudy images
    if is_it_clear != True:

        # If image is cloudy, records as 'cloudy'
        processed_images.write(str(file) + ' cloudy' + '\n')
        processed_images.close()
        streaks_file.close()
        return

    # Estimate signal, noise and greyscale image
    signal, noise, grey = convert_to_grey(rgb)
    source = np.where(signal < source_extraction_sigmas*np.sqrt(noise), 0, 255)

    # XXX Debugging: store raw source image
    #cv2.imwrite(str(output) + '/detected_streaks/' + file.replace('.NEF', '_source_raw.png'), source)

    # Apply morphological opening to remove noise
    kernel = circular_kernel(opening_kernel_radius)
    source = cv2.morphologyEx(source.astype('uint8'), cv2.MORPH_OPEN, kernel)

    # Apply morphological closing to merge streaks that are fragmented
    kernel = circular_kernel(closing_kernel_radius)
    source = cv2.morphologyEx(source.astype('uint8'), cv2.MORPH_CLOSE, kernel)

    # XXX Debugging: store processed source image
    #cv2.imwrite(str(output) + '/detected_streaks/' + file.replace('.NEF', '_source_processed.png'), source)

    # Connect pixels to build sources.
    # See https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(source.astype('uint8'), connectivity)

    # Debugging: render extracted sources
    #print('Found ' + str(num_labels) + ' sources')
    # Map component labels to hue val
    #label_hue = np.uint8(179*labels/np.max(labels))
    #blank_ch = 255*np.ones_like(label_hue)
    #labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    #labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    #labeled_img[label_hue==0] = 0
    #cv2.imwrite(str(output) + '/detected_streaks/' + file.replace('.NEF', '_labels.png'), labeled_img)

    # Empty container to store extracted streaks
    streaks = [None] * 0

    # Iterate over the sources and look for streaks
    for i in range(1,num_labels):

        stat = stats[i]
        size = stat[cv2.CC_STAT_AREA]

        if size < sizethresh:
            continue

        # Get the indices of the pixels comprising this source; iterate over just the bounding box for efficiency
        pixels = [None] * 0
        for x in range(stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH]):
            for y in range(stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]):
                if labels[y,x] == i:
               	    pixels.append((x,y))

        length, width, a, b = detect_streak(pixels)

        # The ratio of length to width determines if this is a streak or not
        ratio = length / width

        if ratio > streak_aspect_ratio_min and width > streak_width_min:
            # Found a streak! Get the end points, subtracting half the width from
            # each end point to correct for PSF size.
            streaks.append([a[0], a[1], b[0], b[1]])

    # Convert streaks list to array
    streaks = np.asarray(streaks)

    print('Found ' + str(streaks.shape[0]) + ' streaks')

    if len(streaks) == 0:
        # Image is clear but no streaks found
        processed_images.write(str(file) + ' clear_streakless' + '\n')
        processed_images.close()
        streaks_file.close()
        return

    # XXX Debugging: draw lines onto original image
    #for x1, y1, x2, y2 in streaks:
    #    cv2.line(rgb, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
    #cv2.imwrite(str(output) + '/detected_streaks/' + file.replace('.NEF', '_lines.png'),rgb)

    # Process each detected streak separately assuming they are different satellites
    for idx, streak in enumerate(streaks):

        x1, y1, x2, y2 = streak

        # Note - by convention the origin of the image coordinates is at the upper left corner;
        # x increases to the right and y increases down.
        centre_xcoordinate = int(x1 + (x2-x1)/2.0)
        centre_ycoordinate = int(y1 + (y2-y1)/2.0)

        # Image width & height, including margin if necessary
        width = int(max(thumbnail_min_diameter, 2*thumbnail_streak_margin + np.abs(x1-x2)))
        height = int(max(thumbnail_min_diameter, 2*thumbnail_streak_margin + np.abs(y1-y2)))
            
        # Image boundaries, clamped to edges of the full image
        x_lo = max(0, centre_xcoordinate - int(width/2))
        x_hi = min(centre_xcoordinate + int(width/2), len(grey[1]) - 1)
        y_lo = max(0, centre_ycoordinate - int(height/2))
        y_hi = min(centre_ycoordinate + int(height/2), len(grey[0]) - 1)

        # Extract thumbnail image surrounding streak.
        streak_image = grey[y_lo : y_hi, x_lo : x_hi]

        # Draw streak into thumbnail image
        # cv2.line(streak_image, (int(x1 - x_lo),int(y1 - y_lo)), (int(x2 - x_lo),int(y2 - y_lo)), (0,0,255), 2)
            	
        # Save thumbnail to disk
        streak_filepath = str(output) + '/detected_streaks/' + file.replace('.NEF', '_streak_' + str(idx+1) + '.png')
        cv2.imwrite(streak_filepath, streak_image)

        # Location for WCS output from Astrometry.NET
        wcsfile = str(output) + '/wcs/' + file.replace('.NEF', '_wcs_' + str(idx+1) + '.fits')

        # Compose Astrometry.NET command and run it synchronously (wait for results)
        cmd = '%s %s --apikey %s --upload %s --wcs %s' % (pythonpath, clientpath, apikey, streak_filepath, wcsfile)
        os.system(cmd)

        # Load the WCS & extract the calibration info from the header.
        # Note: Throws a warning that the axes of the WCS file are 0 
        # when the expected number of axes is 2. This can be ignored,
        # the program will continue running.
        hdu = fits.open(wcsfile)
        w = WCS(hdu[0].header)
            
        # Transform pixel coordinates of streak end points to celestial coordinates.
        # Remember to translate streak coordinates to thumbnail image frame.
        ra1, dec1 = w.wcs_pix2world(x1 - x_lo, y1 - y_lo, 0, ra_dec_order=True)
        ra2, dec2 = w.wcs_pix2world(x2 - x_lo, y2 - y_lo, 0, ra_dec_order=True)

        # Extracts timestamp from filename and converts into a datetime object
        filename_list = list(str(file))
        timestamp = "".join(filename_list[15:21])
        time = datetime.datetime.strptime(timestamp, '%H%M%S')

        # Sets the times for the streak endpoints to be the image
        # timestamp and the timestamp + shutter speed.
        # TODO: Make exposure time a global parameter
        time_a = time.time()
        time_b = (time + datetime.timedelta(seconds=5)).time()
            
        # Write details of streak to the file
        streaks_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (file, ra1, dec1, x1, y1, ra2, dec2, x2, y2, time_a, time_b))                    

    # Record image as clear_streak, with number of streaks
    processed_images.write(str(file) + ' clear_streak ' + str(len(streaks)) + '\n')
    processed_images.close()
    streaks_file.close()

def process_list(filelist, output):

    # TODO: don't keep opening and closing the file streams. Keep them open until
    # finished then close them all.
    # TODO: rationalise this main loop

    for file in filelist:
    # The first loop processes all images in a directory and returns a text file
    # containing streak data. The data consists of the filename of
    # an image containing a satellite streak, together with coordinate and
    # timestamp information used in the next step to calculate orbits.
        print(file)
        
        streaks = open(output + 'streaks_data.txt','a+')
        # Creates a .txt document to store data extracted from image processing loop
        processed_images = open(output + 'processed_images.txt', 'a+')
        # Creates a .txt document to store filenames of images as they are processed
        processed_images_read = open(output + 'processed_images.txt', 'r')   
        # Read-only version of processing record
        already_processed = processed_images_read.read().split()
        
        if file in already_processed:   
            processed_images.close()
            processed_images_read.close()
            streaks.close()
            print('    already processed')
            continue
            # Skips already processed files and continues to next iteration
            
        else: 
            process_image(datadirectory, file, streaks, processed_images, output)
            
if __name__ == "__main__":                
    filelist = os.listdir(datadirectory)
    process_list(filelist, output)
