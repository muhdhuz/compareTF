import numpy as np
from PIL import Image

import json

#------------------------------------------------
# For reading/writing single-channel tiff spectrograms in [a,b] frequency scale
# Also able to scale spectrograms to desired size
#------------------------------------------------

def scale_img(img, basewidth, baseheight):
    """
    Rescale images according to specified basewidth / baseheight in terms of pixels. 
    Maintains aspect ratio if only basewidth is specified.
    """
    if (baseheight == None):        
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    else:
        img = img.resize((basewidth, baseheight), Image.ANTIALIAS)
    return img


def PNG2LogSpect(fname):
    """
    Read tiff spectrograms, and expand to original scale and return numpy array.
    The values needed to undo previous scaling are stored in one of the tiff headers.
    """
    img = Image.open(fname)
    #print("img.info is " + str(img.info))
    minx=-80.
    maxx=0.
    a=0.
    b=255.

    outimg = np.asarray(img, dtype=np.float32)
    outimg = (outimg )/(b)*(-minx) + minx
    return np.flipud(outimg)


def logSpect2PNG(outimg, fname, neww=None, newh=None, info=None) :
    
    if neww is not None:
        savimg = Image.fromarray(outimg)
        outimg = scale_img(savimg,neww,newh)
    
    shift = np.amax(outimg) - np.amin(outimg)
    SC2 = 255*(outimg-np.amin(outimg))/shift
    savimg2 = Image.fromarray(np.flipud(SC2))

    pngimg = savimg2.convert('L')  
    #if (info != None) :  
    #    pngimg.info=json.dumps(info)
    pngimg.save(fname )

    

def PNG2MagSpect(fname) :
    logmag  = PNG2LogSpect(fname)
    return (np.power(10, logmag/20.) )

