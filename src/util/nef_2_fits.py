"""
Python utility for conversion of raw NEF RGB image files to FITS files separated
by R/G/B channel.
"""
import os, rawpy, cv2
from astropy.io import fits
import numpy as np

# Globals
rgb = ['red', 'green', 'blue']

def nef_2_fits(input, output):

    # Optionally apply vignet correction
    vignet_r_path = '/home/nrowell/Projects/SSA/FireOPAL/flatfield/output/vignet_r.png'
    vignet_g_path = '/home/nrowell/Projects/SSA/FireOPAL/flatfield/output/vignet_g.png'
    vignet_b_path = '/home/nrowell/Projects/SSA/FireOPAL/flatfield/output/vignet_b.png'

    # Load vignetting corrections; ndarrays of shape (4928, 7380)
    vignet_r = cv2.imread(vignet_r_path, cv2.IMREAD_GRAYSCALE)
    vignet_g = cv2.imread(vignet_g_path, cv2.IMREAD_GRAYSCALE)
    vignet_b = cv2.imread(vignet_b_path, cv2.IMREAD_GRAYSCALE)

    # Scale vignetting correction images to [0:1] range
    #vignet_r = np.divide(vignet_r, 255)
    #vignet_g = np.divide(vignet_g, 255)
    #vignet_b = np.divide(vignet_b, 255)

    #vignet_r = np.divide(vignet_r, 223)
    #vignet_g = np.divide(vignet_g, 223)
    #vignet_b = np.divide(vignet_b, 223)

    vignet_r = np.divide(vignet_r, 191)
    vignet_g = np.divide(vignet_g, 191)
    vignet_b = np.divide(vignet_b, 191)

    # Size of vignet correction images
    VIGNET_WIDTH = 7380
    VIGNET_HEIGHT = 4928

    # Size of images to be corrected
    NEF_WIDTH = 7380
    NEF_HEIGHT = 4928

    vignet = np.empty([NEF_HEIGHT, NEF_WIDTH, 3])

    print('Creating merged & reshaped vignetting correction image')

    for i in range(0, NEF_HEIGHT):
        x = round(((VIGNET_HEIGHT-1) * i) / (NEF_HEIGHT-1))
        for j in range(0, NEF_WIDTH):
            y = round(((VIGNET_WIDTH-1) * j) / (NEF_WIDTH-1))
            vignet[i,j,0] = vignet_r[x,y]
            vignet[i,j,1] = vignet_b[x,y]
            vignet[i,j,2] = vignet_g[x,y]


    # Locate all files with NEF extension
    filelist = [f for f in os.listdir(input) if f.endswith('.NEF')]

    for file in filelist:

        print('Processing file ', file)

        # Read raw NEF image
        raw = rawpy.imread(input + '/' + file)

        # Convert to standard RGB pixels [0:255]
        # Note that rawpy offers control over how the demosaicing is done, e.g.
        # rgbimage = raw.postprocess(gamma=(1,1), no_auto_bright=True)
        # The default settings seem to work OK, but it's uncertain how close
        # the resulting pixel values are to the raw photoelectron counts.

        # numpy.ndarray of shape (4928, 7380, 3)
        rgbimage = raw.postprocess(output_bps=16)

        # Vignetting corrected version
        rgbimage = np.divide(rgbimage, vignet)
        rgbimage = np.clip(rgbimage, 0, 255)

        # Process each R/G/B channel in turn
        for i in range(3):

            channel = rgb[i]

            print(' - channel: ', channel)

            # numpy.ndarray of shape (4928, 7380)
            image = rgbimage[:,:,i]

            # Filename
            path = str(output) + '/' + file.replace('.NEF', '_' + channel + '.fits')

            # Creates PrimaryHDU object
            hdu = fits.PrimaryHDU(data=image)

            # Add metadata to header
            hdu.header['SOURCE'] = 'Royal Observatory Edinburgh satellite camera'
            hdu.header['CHANNEL'] = channel

            # Write to FITS image using astropy; overwrite any existing file
            hdu.writeto(path, overwrite=True)

if __name__ == "__main__":
    input = '/home/nrowell/Projects/SeniorHonoursProjects/2022-2023/images/2023-02-13/NEF'
    output = '/home/nrowell/Projects/SeniorHonoursProjects/2022-2023/images/2023-02-13/FITS_VIGNET_CORRECTED'
    nef_2_fits(input, output)
