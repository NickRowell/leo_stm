"""
Image processing application for generation of RGB flatfield images
from cloudy images.

"""
import os, rawpy, cv2
import numpy as np

# Global settings
MAX_SATURATION_PERCENT = 10

def create_flatfield(input, output):

    filelist = os.listdir(input)
    
    rgb = ['r', 'g', 'b']
    
    # Do each channel separately to reduce memory footprint
    for i in range(3):
    
        channel = rgb[i]
        
        # Array to store selected & preprocessed RGB frames
        frames = np.empty([20, 4928, 7380], dtype=np.uint8)
        
        idx = 0
    
        for file in filelist:
    
            print('Processing file ', file, ' channel: ', channel)        
    
            # Read raw NEF image
            raw = rawpy.imread(input + '/' + file)
    
            # Convert to standard RGB pixels [0:255]
            # Note that rawpy offers control over how the demosaicing is done, e.g.
            # rgbimage = raw.postprocess(gamma=(1,1), no_auto_bright=True)
            # ...however the method we use here should match what's used in the
            # image processing code, to make the flatfield consistent.
            
            # numpy.ndarray of shape (4928, 7380, 3)
            rgbimage = raw.postprocess()
    
            # numpy.ndarray of shape (4928, 7380)
            image = rgbimage[:,:,i]
            
            # Detect saturated images; count number of elements that are 255
            sat = (image == 255).sum()
            sat_perc = 100 * sat / (4928 * 7380)
            
            if sat_perc < MAX_SATURATION_PERCENT:
                frames[idx] = preprocess(image)
                path = str(output) + '/' + file.replace('.NEF', '_' + channel + '.png')
                cv2.imwrite(path, frames[idx])
                idx += 1
    
            print('Used frames = ', idx)
            
            
        # Got all preprocessed images; construct flatfield
        flatfield = construct_flatfield(frames)
        
        # Write flatfields to file
        path = str(output) + '/' + 'vignet_' + channel + '.png'
        
        # Written as PNG image data, 7380 x 4928, 8-bit grayscale, non-interlaced
        cv2.imwrite(path, flatfield)
    
def preprocess(frame):
    
    """
    Prepares a single R/G/B image for use in flatfield estimation by smoothing,
    normalising etc.
    """
    
    # frame pixels -> numpy.uint8
    
    # Measure backgrounds of each image
    frame_smooth = cv2.medianBlur(frame, 71)
    
    # frame_smooth pixels -> numpy.uint8
    
    # Normalise to average pixel value (causes above-average pixels to be above 1.0)
    #norm = np.mean(frame_smooth)
    
    # Normalise to max pixel value
    #norm =  np.max(frame_smooth)
    
    # Normalise to 99th percentile level to make robust to noise
    norm = np.percentile(frame_smooth, 99)
    
    # Normalise
    frame_smooth = np.divide(frame_smooth, norm/255)
    
    # Clip values to uint8 range
    frame_smooth = np.clip(frame_smooth, 0, 255)
    
    # Convert back to uint8 to save memory
    frame_smooth = frame_smooth.astype(np.uint8)
    
    return frame_smooth

def construct_flatfield(frames_array):
    
    # Take unweighted median of the individual frames
    median = np.median(frames_array, axis=0)
    
    # Normalise the median image
    norm = np.percentile(median, 99)
    
    median = np.divide(median, norm/255.0)
    
    return median


if __name__ == "__main__":
    input = '/home/nrowell/Projects/SSA/FireOPAL/flatfield/input/NEF'
    output = '/home/nrowell/Projects/SSA/FireOPAL/flatfield/output'
    create_flatfield(input, output)
