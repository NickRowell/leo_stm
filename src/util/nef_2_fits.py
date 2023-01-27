"""
Image processing application for generation of RGB flatfield images
from cloudy images.

"""
import os, rawpy
from astropy.io import fits

# Globals
rgb = ['red', 'green', 'blue']

def nef_2_fits(input, output):

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
        rgbimage = raw.postprocess()

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
    input = '/home/nrowell/Projects/SeniorHonoursProjects/2022-2023/images/NEF'
    output = '/home/nrowell/Projects/SeniorHonoursProjects/2022-2023/images/FITS'
    nef_2_fits(input, output)
