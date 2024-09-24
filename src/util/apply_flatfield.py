"""
Image processing application used to apply an existing flatfield/vignet correction
to JPEG images, primarily for use in creating better visualisations.

"""
import os, cv2
import numpy as np


def process_jpgs(input, output):

    VIGNET_WIDTH = 7380
    VIGNET_HEIGHT = 4928

    # Path to vignetting correction images
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
    
    # Merge and reshape to (JPG_WIDTH, JPG_HEIGHT, 3)
    
    # preview 4 resolution:
    #JPG_WIDTH = 7360
    #JPG_HEIGHT = 4912
    
    # preview 3 resolution:
    JPG_WIDTH = 1620
    JPG_HEIGHT = 1080
    
    vignet = np.empty([JPG_HEIGHT, JPG_WIDTH, 3])
    
    print('Creating merged & reshaped vignetting correction image')
    
    for i in range(0, JPG_HEIGHT):
        x = round(((VIGNET_HEIGHT-1) * i) / (JPG_HEIGHT-1))
        for j in range(0, JPG_WIDTH):
            y = round(((VIGNET_WIDTH-1) * j) / (JPG_WIDTH-1))
            vignet[i,j,0] = vignet_r[x,y]
            vignet[i,j,1] = vignet_b[x,y]
            vignet[i,j,2] = vignet_g[x,y]
    
    print('Finished creating vignetting correction image')
    
    filelist = os.listdir(input)
    
    for file in filelist:

        print('Processing file ', file)        
        
        # Read JPG image
        
        # ndarray of shape (4912, 7360, 3)
        raw =  cv2.imread(input + '/' + file)
        
        # Vignetting corrected version
        proc = np.divide(raw, vignet)
        proc = np.clip(proc, 0, 255)
        
        # To change filename ending:
        # file.replace('.jpg', '_corr.jpg')
        path = str(output) + '/' + file
        cv2.imwrite(path, proc)

if __name__ == "__main__":
    input = '/home/nrowell/Projects/SSA/FireOPAL/videos/2022-05-11/preview3/JPG_RAW'
    output = '/home/nrowell/Projects/SSA/FireOPAL/videos/2022-05-11/preview3/JPG_PROCESSED'
    process_jpgs(input, output)
