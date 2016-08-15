from det_out_base import DetOutBase

from larcv import larcv
larcv.load_pyutil


class DetOutLarcv(DetOutBase):
    
    def __init__(self,init_data):
        super(DetOutBase,self).__init__()
        
        self.name="DetOutLarcv"
        self.outfname=init_data['outfname']
        
        self.iniom  = init_data['iniom']
        self.outiom = larcv.IOManager(larcv.IOManager.kWRITE)
        self.outiom.set_out_file(self.outfname)
        self.outiom.initialize()
        
        self.copy_input=init_data['copy_input']
        
    def roi2imgcord(self,imm,pos):
        
        x = pos[0]
        y = pos[1]
        
        # the scale factor
        dw_i = imm.cols() / (imm.max_x() - imm.min_x())
        dh_i = imm.rows() / (imm.max_y() - imm.min_y())
        
        # convert into image coordinates
        x /= dw_i
        y /= dh_i
        
        # the origin
        origin_x = x + imm.min_x()
        origin_y = y + imm.min_y()
        
        # w_b is width of a rectangle in original unit
        w_b = pos[2]-pos[0]
        h_b = pos[3]-pos[1]
        
        # the width
        width  = w_b / dw_i
        height = h_b / dh_i

        # for now fill w/ 0
        row_count = 0
        col_count = 0
        
        # vic isn't sure why this is needed for imshow to larcv
        origin_y += height
    
        return (width,height,row_count,col_count,origin_x,origin_y)

    def __write_event__(self,event_data):

        out_evroi = out_iom.get_data(larcv.kProductROI,"valid")
        out_roi = larcv.ROI()
    
        in_image = in_iom.get_data(larcv.kProductImage2D,"tpc")
        in_roi   = in_iom.get_data(larcv.kProductROI,"tpc")
    
        in_image_meta=in_image.Image2DArray()[2].meta()
        
        for bbox in event_data['bboxes']:
            
            xx1=bbox['box'][0]
            yy1=bbox['box'][1]
            xx2=bbox['box'][2]
            yy2=bbox['box'][3]
            
            #awkward way to convert caffe to larcv
            y2= imm.rows()-xx1
            x1= yy1
            y1= imm.rows()-xx2
            x2= yy2
        
            pos = [x1,y1,x2,y2]
            
            width,height,row_count,col_count,origin_x,origin_y = roi2imgcord(in_image_meta,pos)
        
            bbox_meta = larcv.ImageMeta(width,height,
                                        row_count,col_count,
                                        origin_x,origin_y,2)
            
            
            #WARNING: neet to set bbox['cls'] to ROITYPE!
            
            out_roi.AppendBB(larcv.ImageMeta(0,0,0,0,in_image_meta.min_x(),in_image_meta.max_y(),0))
            out_roi.AppendBB(larcv.ImageMeta(0,0,0,0,in_image_meta.min_x(),in_image_meta.max_y(),1))
            out_roi.AppendBB(bbox_meta)

            out_roi.NetworkProb( float(bbox['prob']) )
            out_evroi.Append(out_roi)
        
        
        if self.copy_input == True:
            
            copy_img = out_iom.get_data(larcv.kProductImage2D,"tpc")
            copy_roi = out_iom.get_data(larcv.kProductROI,"tpc")

            for img in in_image.Image2DArray(): copy_img.Append(img)   
        
            copy_roi.Set(in_roi.ROIArray())

        out_iom.set_id(in_image.run(),in_image.subrun(),in_image.event())       
        out_iom.save_entry()
        
    def __close__(self):
        self.outiom.finalize()
        
