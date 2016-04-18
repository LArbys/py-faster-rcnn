from ROOT import larcv
larcv.load_pycvutil
import numpy as np

from fast_rcnn.config import cfg
from utils.blob import im_list_to_blob

FILES = cfg.ROOTFILES

IMAGE2DPROD = cfg.IMAGE2DPROD
ROIPROD     = cfg.ROIPROD

#Load 

print "'\033[92m'\t>> IOManager Loading in root_handler.py\n'\033[0m'"
IOM = larcv.IOManager(larcv.IOManager.kREAD)

for F in FILES:
    IOM.add_in_file(F)
    
#IOM.set_verbosity(0)

print "'\033[94m'\t>> initialize IOManager \n'\033[0m'"
IOM.initialize()

def get_n_images() :
    return IOM.get_n_entries()

# def get_roi_data(entry) :
    
#     IOM.read_entry(entry)
#     ev_roi = IOM.get_data(larcv.kProductROI,ROIPROD)

#     #plane == 0
#     return larcv.split_ev_rois(ev_roi,0)


def get_im_blob(roidb,scale_inds) :
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []

    for i in xrange(num_images):
        #im = cv2.imread(roidb[i]['image'])
        # print roidb[i]
        
        IOM.read_entry( int( roidb[i]['image'] ) )
        ev_img = IOM.get_data(larcv.kProductImage2D,IMAGE2DPROD)

        im  = larcv.as_ndarray( ev_img.Image2DArray()[0] )
        s   = im.shape
        imm = np.zeros([ s[0], s[1], 3 ])
        
        img_v = ev_img.Image2DArray()

        assert img_v.size() == 3

        for j in xrange(3):
            imm[:,:,j]  = larcv.as_ndarray( img_v[j] )
            imm = imm[:,::-1,:]
            #imm[:,:,i] -= np.mean(imm[:,:,i])
            

        print imm.shape

        if roidb[i]['flipped']:
            imm = imm[:, ::-1, :]
        print im_scales
        im_scales.append(1) #1 to 1 scaling!
        processed_ims.append(imm)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
    
    
    

        


