import os,sys

from larcv import larcv
larcv.load_pyutil
DEBUG=False
CHANNELS=[2]
CLASSES=[5]

if len(sys.argv) < 4:
    print "1=>FILE.root"
    print "2=>N entries (-1) == all entries"
    print "3=>outfile.txt"
    sys.exit(1)

FILE=str(sys.argv[1])
iom=larcv.IOManager()
iom.add_in_file(FILE)
iom.initialize()

nentries=iom.get_n_entries()
if sys.argv[2] != -1:
    nentries=int(sys.argv[2])

print "Got : ",nentries," from input"

f=open(str(sys.argv[3]),'w+')
for index in xrange(nentries):
    
    sys.stdout.write('{} imported\r'.format(index))
    sys.stdout.flush()

    iom.read_entry(index)
    xy = {}

    ev_image = iom.get_data(larcv.kProductImage2D,"comb_tpc")
    ev_roi   = iom.get_data(larcv.kProductROI,"comb_tpc")

    image_v = ev_image.Image2DArray()
    images = [ image_v[k] for k in CHANNELS ]

    roi_v = ev_roi.ROIArray()

    # for each ROI
    for ix,roi in enumerate(roi_v):
        roitype = larcv.PDG2ROIType(roi.PdgCode())
        #is it one of the UB classes?
    
        if roitype not in CLASSES:
            if DEBUG:
                print roitype, "Not in self._classes: ",self._classes
            continue

        if roitype not in xy.keys():
            xy[roitype]=[]            

        for ix,channel in enumerate(CHANNELS):
            #empty ROI?
            if roi.BB().size() == 0:
                continue

            bbox = roi.BB(channel)
            
            imm = images[ix].meta()
            
            x = bbox.bl().x - imm.bl().x
            y = bbox.bl().y - imm.bl().y
            
            dw_i = imm.cols() / ( imm.tr().x - imm.bl().x )
            dh_i = imm.rows() / ( imm.tr().y - imm.bl().y )
            
            w_b = bbox.tr().x - bbox.bl().x
            h_b = bbox.tr().y - bbox.bl().y
            
            x1 = x*dw_i
            x2 = (x + w_b)*dw_i
            
            y1 = y*dh_i
            y2 = (y + h_b)*dh_i
            
            assert x2>x1
            assert y2>y1
            
            imrows=imm.rows()
            imcols=imm.cols()

            SS=(imrows-y2,x1,imrows-y1,x2)
            if SS in xy[roitype] : 
                SS=()
                continue
            xy[roitype].append(SS)
            f.write("{} {} {} {} {} {}\n".format(index,roitype,SS[0],SS[1],SS[2],SS[3]))
f.close()
