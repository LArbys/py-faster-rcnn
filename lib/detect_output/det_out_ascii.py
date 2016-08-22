from det_out_base import DetOutBase


class DetOutAscii(DetOutBase):
    
    def __init__(self,init_data):
        super(DetOutBase,self).__init__()
        
        self.name="DetOutAscii"
        self.outfname=init_data['outfname']
        self.outfile = fopen()

    def __write_event__(self,event_data):
        raise Exception()
        
    def __close__(self):
        self.outfile.close()
