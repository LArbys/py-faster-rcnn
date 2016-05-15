from imageloaderbase import ImageLoaderBase

class LarbysDetectLoader(ImageLoaderBase):

    def __init__(self):
        super(LarbysDetectLoader,self).__init__()
        self.name = "LarbysDetectLoader"

    def __load_image__(self,img):

        for i in xrange(3):
            img[:,:,i] = img[:,:,i].T
        
        img = img[::-1,:,:]

        return img
        
        
