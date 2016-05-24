from imageloaderbase import ImageLoaderBase

class BNBNuv04Loader(ImageLoaderBase):

    def __init__(self,imin,imax):
        super(BNBNuv04Loader,self).__init__()
        self.name = "BNBNuv04Loader"
        self.imin = imin
        self.imax = imax

    def __load_image__(self,img):
        
        assert img.shape[-1] == 3 #better be three channels

        # subtract off the minimum
        
        scale = 100.0

        img *= scale

        img -= self.imin * scale;
        
        # don't allow negatives
        img[ img < 0  ] = 0

        # cut the top off
        img[ img > self.imax*scale] = self.imax*scale
    
        img = img[:,:,2:]
        
        #img = img[:,:,2:]
        
        # make sure we only return a single channel
        assert img.shape[-1] == 1
        
        return img
        
        
