import abc

class ImageLoaderBase(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.name = "ImageLoaderBase"

    def load_image(self,img):
        return self.__load_image__(img)

    @abc.abstractmethod
    def __load_image__(self,img):
        """ Transform the image in a specific way """
        
