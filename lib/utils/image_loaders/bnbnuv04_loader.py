from imageloaderbase import ImageLoaderBase

class BNBNuv04Loader(ImageLoaderBase):

    def __init__(self,cfg):
        super(BNBNuv04Loader,self).__init__()
        self.name = "BNBNuv04Loader"
        self.imin = cfg.IMIN
        self.imax = cfg.IMAX
        self.scale= cfg.SCALE

    def __load_image__(self,img):
        
        # assert img.shape[-1] == 3 #better be three channels

        # subtract off the minimum
        
        img *= self.scale

        img -= self.imin * self.scale;
        
        # don't allow negatives
        img[ img < 0  ] = 0

        # cut the top off
        img[ img > self.imax*self.scale] = self.imax*self.scale
    
        # img = img[:,:,2:]
        
        # make sure we only return a single channel
        assert img.shape[-1] == 1
        
        return img
        
        
