from imageloaderbase import ImageLoaderBase

class MergedLoader(ImageLoaderBase):

    def __init__(self):
        super(MergedLoader,self).__init__()
        self.name = "MergedLoader"

    def __load_image__(self,img):
        
        assert img.shape[-1] == 3 #better be three channels

        for i in xrange(3):
            img[:,:,i] = img[:,:,i].T
        
        img[ img < 0 ]   = 0
        img[ img > 256 ] = 256
    
        img = img[::-1,:,:]

        return img
        
        
