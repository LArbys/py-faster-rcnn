from imageloaderbase import ImageLoaderBase

class BNBNuLoader(ImageLoaderBase):

    def __init__(self,imin,imax):
        super(BNBNuLoader,self).__init__()
        self.name = "BNBNuLoader"
        self.imin = imin
        self.imax = imax

    def __load_image__(self,img):
        
        assert img.shape[-1] == 3 #better be three channels

        # subtract off the minimum
        img -= self.imin;
        
        # don't allow negativees
        img[ img < 0  ] = 0

        # cut the top off
        img[ img > self.imax ] = self.imax
    
        img = img[::-1,:,2:]
        
        assert img.shape[-1] == 1
        
        return img
        
        
