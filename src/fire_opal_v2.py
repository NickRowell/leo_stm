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

def convert_to_grey(rgbimage):
    
    """ This function converts an RGB image to a 
    greyscale image by averaging the RGB values.
    
    Input: RGB image
    Output: Greyscale image """

    greyimage = np.sum(rgbimage, axis=2)/(3*255.0)
    return greyimage

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
    
def normal_line(x1, y1, x2, y2):

    """
    Calculates the normal representation of the line from two points.
    Can handle vertical lines.

    Inputs: two sets of (x,y) pixel coordinates
    Outputs: orientation (theta [rad]) and distance to origin (d)

    The orientation angle is measured anticlockwise from the x axis
    and lies in the range [-pi:pi]. The distance to origin is always
    positive.
    
    """

    dx = x2-x1
    dy = y2-y1
    n = np.sqrt(dx*dx + dy*dy)

    # Compute the unit normal to the line. At this stage it may point
    # towards or away from the origin.
    nx = dy/n
    ny = -dx/n

    # Compute the distance to the origin
    d = x1 * nx + y1 * ny

    # If this is negative then flip the direction of the normal
    if d < 0:
        nx *= -1
        ny *= -1
        d *= -1

    # Absolute value of angle measured from x axis to the normal
    theta = np.arccos(nx)

    # Correct sign
    if ny < 0:
        theta *= -1

    return theta, d

def line_from_two_points(x1, y1, x2, y2):

    """
    This function calculates the slope and intercept of a line from two points.
    
    Inputs: two sets of (x,y) pixel coordinates
    Outputs: slope m, intercept b
    
    """
    if x2 == x1: 
        x2 += 1
    if y2 == y1:
        y2 += 1
    # Avoids NAN or 0 slope errors
    
    m = float(y2-y1)/float(x2-x1)
    b = (y2 - m*x2)
    return m, b

def process_image(datadirectory, file, streaks, processed_images, processed_images_read, output):

    # Read raw NEF image
    raw = rawpy.imread(datadirectory + file)

    # Convert to standard RGB pixels [0:255]
    rgb = raw.postprocess()

    # Convert to greyscale pixels [0:1]
    greyscale_image = convert_to_grey(rgb)

    # Returns True if the image is clear and False if cloudy
    is_it_clear = cloudy_or_clear(greyscale_image)
    
    # TODO simplify this conditional
    if is_it_clear == True:

        # Denormalise image
        greyscale_image *= 255.0

        # Estimate the background image
        # bkg = cv2.medianBlur(greyscale_image.astype('uint8'), 71)

        # Background subtraction
        # greyscale_image -= bkg

        # First a Canny edge detector creates a binary black and white image
        # in which edges are shown in white.
        edges = cv2.Canny(greyscale_image.astype('uint8'), definitely_not_an_edge, definitely_an_edge, apertureSize=3)

        # Write edge detection image to file
        # cv2.imwrite(str(output) + '/detected_streaks/edges.png', edges)

        # Perform probabilistic Hough transform to find lines in image.
        # Returns array of the form [[[x1 y1 x2 y2]], [[x1 y1 x2 y2]], ... ]
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, line_votes, minLineLength=50, maxLineGap=5)


        # Draw lines onto original image
        # Iterates over [[x1 y1 x2 y2]], [[x1 y1 x2 y2]], ...
        # for line in lines:
            # From [[x1 y1 x2 y2]] extract [x1 y1 x2 y2]
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(rgb, (x1,y1), (x2,y2), (0,0,255), 2)
        # cv2.imwrite(str(output) + '/detected_streaks/lines.png',rgb)

        if lines is not None: 
            
            # More than one streak - MIGHT NOT BE SAFE TO AVERAGE END POINTS!
            # We're assuming these are multiple lines corresponding to the same streak,
            # but they might not be if there's more than one streak!
            if lines.shape[0] > 1:
                
                # If the satellite streak has a width greater than one pixel,
                # HoughLineP will interpret it as multiple lines placed very
                # close together. We solve this by averaging the endpoints of
                # the lines to get an estimate of the 'real' endpoint.

                #x1 = np.mean(lines[: lines.shape[0],0,0].tolist(), dtype=np.float64)
                #y1 = np.mean(lines[: lines.shape[0],0,1].tolist(), dtype=np.float64)
                #x2 = np.mean(lines[: lines.shape[0],0,2].tolist(), dtype=np.float64)
                #y2 = np.mean(lines[: lines.shape[0],0,3].tolist(), dtype=np.float64)

                # NR: to mitigate the risk of averaging lines that actually correspond
                # to different streaks, just use the first line

                # First index iterates over the line number, third index iterates over the end
                # point coordinates
                x1 = float(lines[0][0][0])
                y1 = float(lines[0][0][1])
                x2 = float(lines[0][0][2])
                y2 = float(lines[0][0][3])
               
            elif lines.shape[0] == 1:
                
                # If HoughLinesP returns only one line, no further processing
                # of the endpoints is needed.
                
                x1 = float(lines[:,0,0])
                y1 = float(lines[:,0,1])
                x2 = float(lines[:,0,2])
                y2 = float(lines[:,0,3])
            
            # Note - by convention the origin of the image coordinates is at the upper left corner;
            # x increases to the right and y increases down.
            centre_xcoordinate = int(x1 + (x2-x1)/2.0)
            centre_ycoordinate = int(y1 + (y2-y1)/2.0)

            # Image width & height, including margin if necessary
            width = int(max(thumbnail_min_diameter, 2*thumbnail_streak_margin + np.abs(x1-x2)))
            height = int(max(thumbnail_min_diameter, 2*thumbnail_streak_margin + np.abs(y1-y2)))
            
            # Image boundaries, clamped to edges of the full image
            x_lo = max(0, centre_xcoordinate - int(width/2))
            x_hi = min(centre_xcoordinate + int(width/2), len(greyscale_image[1]) - 1)
            y_lo = max(0, centre_ycoordinate - int(height/2))
            y_hi = min(centre_ycoordinate + int(height/2), len(greyscale_image[0]) - 1)

            # Extract thumbnail image surrounding streak.
            streak_image = greyscale_image[y_lo : y_hi, x_lo : x_hi]

            # Save thumbnail to disk
            streak_filepath = str(output) + '/detected_streaks/' + file.replace('.NEF', '_streak.png')
            cv2.imwrite(streak_filepath, streak_image)

            # Location for WCS output from Astrometry.NET
            wcsfile = str(output) + '/wcs/' + file.replace('.NEF', '_wcs.fits')

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

            # Compute line parameters for the streak. These are in the original image frame,
            # for consistency across image sequences.
            slope, intercept = line_from_two_points(x1, y1, x2, y2)
            
            # Extracts timestamp from filename and converts into a datetime object
            filename_list = list(str(file))
            timestamp = "".join(filename_list[15:21])
            time = datetime.datetime.strptime(timestamp, '%H%M%S')

            # Sets the times for the streak endpoints to be the image
            # timestamp and the timestamp + shutter speed.
            # TODO: Make exposure time a global parameter
            endpointa_time = time.time()
            endpointb_time = (time + datetime.timedelta(seconds=5)).time()
            
            # Writes all extracted data to a .txt file and records
            # as 'clear_streak'
            streaks.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (file, timestamp, ra1, dec1, x1, y1, ra2, dec2, x2, y2, endpointa_time, endpointb_time, slope, intercept))                    
            processed_images.write(str(file) + ' clear_streak' + '\n')
            processed_images.close()
            processed_images_read.close()
            streaks.close()
            
        elif lines == None:
            
            # If image is clear but has no streaks, records as
            # 'clear_streakless'
            processed_images.write(str(file) + ' clear_streakless' + '\n')
            processed_images.close()
            processed_images_read.close()
            streaks.close()
                        
    else:
        
        # If image is cloudy, records as 'cloudy'
        processed_images.write(str(file) + ' cloudy' + '\n')
        processed_images.close()
        processed_images_read.close()
        streaks.close()

def process_list(filelist, output):
    for file in filelist:
    # The first loop processes all images in a directory and returns a text file
    # containing streak data. The data consists of the filename of
    # an image containing a satellite streak, together with coordinate and
    # timestamp information used in the next step to calculate orbits.
        print(file)
        
        streaks = open(output + '/streaks_data.txt','a+')
        # Creates a .txt document to store data extracted from image processing loop
        processed_images = open(output + '/processed_images.txt', 'a+')
        # Creates a .txt document to store filenames of images as they are processed
        processed_images_read = open(output + '/processed_images.txt', 'r')   
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
            process_image(datadirectory, file, streaks, processed_images, processed_images_read, output)
            
            # If the processing throws an error for any reason, the program will
            # write ERROR to the processing record and continue to the
            # next iteration
    #            print(str(e))
    #            processed_images.write(str(file) + ' ERROR' + '\n')
    #            processed_images.close()
    #            processed_images_read.close()
    #            streaks.close()
            
if __name__ == "__main__":                
    filelist = os.listdir(datadirectory)
    process_list(filelist, output)
