from fast_rcnn.config import cfg

from image_loaders.merged_loader        import MergedLoader
from image_loaders.larbys_detect_loader import LarbysDetectLoader

class ImageLoaderFactory(object):
    def __init__(self):
        self.name = "ImageLoaderFactory"
        
    def get(self, loader_name):
        
        if loader_name == "MergedLoader":
            return MergedLoader(cfg.IMIN,cfg.IMAX)
        
        if loader_name == "LarbysDetectLoader":
            return LarbysDetectLoader()
            
        raise Exception("\n\n\t{} does not exist, please instantiate in image_loader_factory.py!\n\n".format(loader_name))
