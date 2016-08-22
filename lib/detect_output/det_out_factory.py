from det_out_larcv import DetOutLarcv
from det_out_ascii import DetOutAscii

class DetOutFactory(object):

    def __init__(self):
        self.name = "DetOutFactory"

    def get(self,loader_name,init_data):
        
        if loader_name == "DetOutLarcv":
            return DetOutLarcv(init_data)

        if loader_name == "DetOutAscii":
            return DetOutAscii(init_data)
            
        raise Exception("\n\n\t{} does not exist, please instantiate in det_out_factory.py!\n\n".format(loader_name))
