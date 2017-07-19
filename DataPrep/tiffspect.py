import numpy as np
from PIL import TiffImagePlugin
from PIL import Image

import json

#------------------------------------------------
# For reading/writing single-channel tiff spectrograms in [a,b] frequency scale
# Also able to scale spectrograms to desired size
#------------------------------------------------

def scale_img(fname, basewidth, baseheight):
    """
    Rescale images according to specified basewidth / baseheight in terms of pixels. 
    Maintains aspect ratio if only basewidth is specified.
    """
    img = Image.open(fname)
    if (baseheight == None):        
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    else:
        img = img.resize((basewidth, baseheight), Image.ANTIALIAS)
    return img

def specScale(outimg, fname, neww, newh, newscale, lwinfo=None):
    """ 
    Single channel spectrogram to tiff file normed to between newscale[0] and newscale[1] 
    """
    info = TiffImagePlugin.ImageFileDirectory_v2()
    lwinfo = lwinfo or {}
    lwinfo['oldmin'] = str(np.amin(outimg))
    lwinfo['oldmax'] = str(np.amax(outimg))
    lwinfo['newmin'] = str(newscale[0])
    lwinfo['newmax'] = str(newscale[1])
        
    info[270] = json.dumps(lwinfo) #info required to reverse scaling
    
    if neww is not None:
        savimg = Image.fromarray(outimg)
        savimg.save(fname, tiffinfo=info)
        outimg = scale_img(fname,neww,newh)
    
    shift = np.amax(outimg) - np.amin(outimg)
    SC = (outimg-np.amin(outimg))/shift
    SC2 = (newscale[1]-newscale[0])*SC + newscale[0]
    savimg2 = Image.fromarray(np.flipud(SC2))
    savimg2.save(fname +'.tif', tiffinfo=info)   
    return SC2

def tif2png(fname, pngname):
    pngimg = Image.open(fname +'.tif').convert('L')
    pngimg.save(pngname +'.png')

def invSpecScale(fname):
    """
    Read tiff spectrograms, and expand to original scale and return numpy array.
    The values needed to undo previous scaling are stored in one of the tiff headers.
    """
    img = Image.open(fname)
    try:
        img.tag[270] = img.tag[270]
    except:
        print('Tiff2LogSpect: no img.tag[270], using default values!')
        lwinfo = {}
        lwinfo['oldmin'] = -80.
        lwinfo['oldmax'] = 0.
        lwinfo['newmin'] = 0
        lwinfo['newmax'] = 255
        img.tag[270] = json.dumps(lwinfo)
    
    lwinfo = json.loads(img.tag[270][0])
    minx, maxx, a, b = float(lwinfo['oldmin']), float(lwinfo['oldmax']), float(lwinfo['newmin']), float(lwinfo['newmax'])
    outimg = np.asarray(img, dtype=np.float32)
    outimg = (outimg - a)/(b-a)*(maxx-minx) + minx
    return np.flipud(outimg), lwinfo

#------------------------------------------------
# Some of Lonce's scaling routines below

def logSpect2Tiff(outimg, fname, lwinfo=None):
    """ 
    Single channel spectrogram to tiff file normed to [0,1] 
    """
    info = TiffImagePlugin.ImageFileDirectory()
            
    scale = 80.
    shift = float(np.amax(outimg))

    lwinfo= lwinfo or {}
    lwinfo["scale"]=scale;
    lwinfo["shift"]=shift;

    #just chose a tag index that appears to be unused: https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
    info[666]=json.dumps(lwinfo)

    #shift to put max at 0
    shiftimg = outimg-shift 
    #scale to map [-80, 0] to  [-1,0]
    outimg = [x / scale for x in outimg]
    #shift to [0,1]
    outimg = [x +1. for x in outimg]
    #clip anything below 0 (anything originally below -80dB)
    outimg = np.maximum(outimg, 0) # clip below 0
    #print('logSpect2Tiff: writing image with min ' + str(np.amin(outimg)) + ', and max ' + str(np.amax(outimg)))
    savimg = Image.fromarray(np.flipud(outimg))
    savimg.save(fname, tiffinfo=info)
    return info[666] # just in case you want it for some reason

def Tiff2LogSpect(fname) :
    """Read tif images, and expand to original scale, return single channel image"""
    img = Image.open(fname)
    #print('Tiff2LogSpect: image min is ' + str(np.amin(img)) + ', and image max is ' + str(np.amax(img)))

    lwinfo=json.loads(img.tag[666][0])
    try:
        scale=lwinfo["scale"]
    except:
        scale=80.

    try:
        shift=lwinfo["shift"]
    except:
        shift=0.

    outimg = np.asarray(img, dtype=np.float32)
    outimg = outimg-1.
    outimg = outimg*scale #  [x *scale for x in outimg] 
    outimg = outimg + shift #  [x +shift for x in outimg] 
    return (np.flipud(outimg), lwinfo)

def Tiff2MagSpect(fname) :
    logmag, lwinfo = Tiff2LogSpect(fname)
    return (np.power(10, logmag/20.) , lwinfo)

