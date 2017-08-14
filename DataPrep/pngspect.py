import numpy as np
from PIL import Image
from PIL import PngImagePlugin
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
    if (basewidth == None):        
        wpercent = (baseheight / float(img.size[1]))
        hsize = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((hsize, baseheight), Image.ANTIALIAS)
    else:
        img = img.resize((basewidth, baseheight), Image.ANTIALIAS)
    return img


def logSpect2PNG(outimg, fname, neww=None, newh=None, lwinfo=None) :
    
    info = PngImagePlugin.PngInfo()
    lwinfo = lwinfo or {}
    lwinfo['oldmin'] = str(np.amin(outimg))
    lwinfo['oldmax'] = str(np.amax(outimg))
    lwinfo['newmin'] = '0'
    lwinfo['newmax'] = '255'
    info.add_text('meta',json.dumps(lwinfo)) #info required to reverse scaling
    
    if newh is not None:
        savimg = Image.fromarray(outimg)
        outimg = scale_img(savimg,neww,newh)
    
    shift = np.amax(outimg) - np.amin(outimg)
    SC2 = 255*(outimg-np.amin(outimg))/shift
    savimg2 = Image.fromarray(np.flipud(SC2))

    pngimg = savimg2.convert('L')  
    pngimg.save(fname,pnginfo=info)
    
def temp(fname):
    img = Image.open(fname)
    print(img.text)
    print(img.text['meta'])

    
def PNG2LogSpect(fname):
    """
    Read tiff spectrograms, and expand to original scale and return numpy array.
    The values needed to undo previous scaling are stored in one of png metadata.
    """
    img = Image.open(fname)
    
    try:
        img.text = img.text
    except:
        print('PNG2LogSpect: no img.text, using default values!')
        lwinfo = {}
        lwinfo['oldmin'] = -80.
        lwinfo['oldmax'] = 0.
        lwinfo['newmin'] = 0
        lwinfo['newmax'] = 255
        info.add_text('meta',json.dumps(lwinfo)) 
    
    lwinfo = json.loads(img.text['meta'])
    minx, maxx, a, b = float(lwinfo['oldmin']), float(lwinfo['oldmax']), float(lwinfo['newmin']), float(lwinfo['newmax'])

    outimg = np.asarray(img, dtype=np.float32)
    outimg = (outimg - a)/(b-a)*(maxx-minx) + minx
    return np.flipud(outimg), lwinfo
   

def PNG2MagSpect(fname) :
    logmag  = PNG2LogSpect(fname)
    return (np.power(10, logmag/20.) )

