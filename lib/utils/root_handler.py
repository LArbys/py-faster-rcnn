from ROOT import larcv
larcv.load_pyutil
import numpy as np

from fast_rcnn.config import cfg
from utils.blob import im_list_to_blob, prep_im_for_blob

from utils.image_loader_factory import ImageLoaderFactory

from easydict import EasyDict as edict


class ROOTHandler(object):

    def __init__(self):
        self.IOCFG = cfg.IOCFG
        self.IMAGE2DPROD = cfg.IMAGE2DPROD
        self.ROIPROD     = cfg.ROIPROD
        
        #Load 
        print "\033[92m\t>> IOManager Loading in root_handler.py\033[0m"
        
        self.IOM = larcv.IOManager(larcv.IOManager.kREAD)

        self.IOM.configure(larcv.CreatePSetFromFile(self.IOCFG))
        #for F in self.FILES:
        #    self.IOM.add_in_file(F)
    
        if cfg.DEBUG : self.IOM.set_verbosity(0)
    
        print "\033[93m\t>> Initializing IOManager\033[0m"
        self.IOM.initialize()

        print "\033[94m\t>> Getting image loader %s\033[0m"%cfg.IMAGE_LOADER
        self.ILF = ImageLoaderFactory()
        self.IMAGELOADER = self.ILF.get(cfg.IMAGE_LOADER)

    def get_n_images(self) :
        return self.IOM.get_n_entries()

    def get_image(self,ttree_index) :
            
        self.IOM.read_entry( ttree_index )
        ev_img = self.IOM.get_data(larcv.kProductImage2D,self.IMAGE2DPROD)
        
        img_v = ev_img.Image2DArray()
        
        im  = larcv.as_ndarray( img_v[0] )
        s   = im.shape
        imm = np.zeros([ s[0], s[1], img_v.size() ])

        # print "just got image shape: {}".format(imm.shape)
            
        assert img_v.size() == 3
    
        for j in xrange(img_v.size()):
            imm[:,:,j]  = larcv.as_ndarray( img_v[j] )
            #img_v[j] *= 30.
        
        return self.IMAGELOADER.load_image(imm)

    def get_im_blob(self,roidb,scale_inds) :
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        processed_ims = []
        im_scales = []
            
        for i in xrange(num_images):

            imm = self.get_image( int( roidb[i]['image'] ) )
            
            assert roidb[i]['flipped'] == False
                
            target_size = cfg.TRAIN.SCALES[0]

            imm, im_scale = prep_im_for_blob(imm, 
                                             cfg.PIXEL_MEANS,
                                             target_size,
                                             cfg.TRAIN.MAX_SIZE)

            im_scale = 1
            im_scales.append(im_scale) #1 to 1 scaling always.
            processed_ims.append(imm)

            # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, im_scales

rh = ROOTHandler()
