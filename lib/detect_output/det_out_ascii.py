from det_out_base import DetOutBase


class DetOutAscii(DetOutBase):
    
    def __init__(self,init_data):
        super(DetOutBase,self).__init__()
        
        self.name="DetOutAscii"
        self.outfname=init_data['outfname']
        self.outfile = open(self.outfname,'w+')

    def __write_event__(self,event_data):
        
        for bbox in event_data['bboxes']:
            
            xx1=bbox['box'][0]
            yy1=bbox['box'][1]
            xx2=bbox['box'][2]
            yy2=bbox['box'][3]
            
            self.outfile.write("{} {} {} {} {} {} {}\n".format(event_data['image_name'],
                                                               bbox['cls'],
                                                               bbox['prob'],
                                                               xx1,yy1,
                                                               xx2,yy2))

    def __close__(self):
        self.outfile.close()
