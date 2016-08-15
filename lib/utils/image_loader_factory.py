from fast_rcnn.config import cfg

from image_loaders.merged_loader        import MergedLoader
from image_loaders.larbys_detect_loader import LarbysDetectLoader
from image_loaders.singlep_loader       import SinglepLoader
from image_loaders.bnbnu_loader         import BNBNuLoader
from image_loaders.bnbnuv04_loader      import BNBNuv04Loader
from image_loaders.bnbnutarinet_loader  import BNBNuTarinetLoader

class ImageLoaderFactory(object):
    def __init__(self):
        self.name = "ImageLoaderFactory"
        
    def get(self, loader_name):
    
        if loader_name == "BNBNuTarinetLoader":
            return BNBNuTarinetLoader(cfg)
        
        if loader_name == "BNBNuv04Loader":
            return BNBNuv04Loader(cfg)

        if loader_name == "BNBNuLoader":
            return BNBNuLoader(cfg)

        if loader_name == "SinglepLoader":
            return SinglepLoader(cfg)
        
        if loader_name == "MergedLoader":
            return MergedLoader(cfg)
        
        if loader_name == "LarbysDetectLoader":
            return LarbysDetectLoader()
            
        raise Exception("\n\n\t{} does not exist, please instantiate in image_loader_factory.py!\n\n".format(loader_name))
