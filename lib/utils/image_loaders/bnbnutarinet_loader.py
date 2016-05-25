from imageloaderbase import ImageLoaderBase

class BNBNuTarinetLoader(ImageLoaderBase):

    def __init__(self,imin,imax):
        super(BNBNuTarinetLoader,self).__init__()
        self.name = "BNBNuTarinetLoader"
        self.imin = imin
        self.imax = imax

    def __load_image__(self,img):
        
        assert img.shape[-1] == 12 #better be twelse channels

        # for i in xrange(3):
        #     img[:,:,i] = img[:,:,i].T


        img -= self.imin

        img[ img < 0 ]   = 0
        img[ img > self.imax ] = self.imax
    
        # img = img[::-1,:,:]

        return img
        
        
