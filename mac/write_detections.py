import _init_paths
from fast_rcnn.config import cfg
import numpy as np
import sys
import caffe, os, sys, cv2

from larcv import larcv
larcv.load_pyutil


if len(sys.argv) < 6:
    print '\n>=================<'
    print 'argv1 = io.cfg'
    print 'arvg2 = prototxt'
    print 'argv3 = caffemodel'
    print 'argv4 = [out_filename].txt'
    print 'argv5 = n examples'
    print '>=================<\n'
    sys.exit(1)

CLASSES = ('__background__', 5)
cfg.NEXAMPLES = sys.argv[5]
cfg.PIXEL_MEANS =  [[[ 0.0 ]]]
cfg.IMAGE2DPROD = "comb_tpc"
cfg.ROIPROD = "comb_tpc"
cfg.WIDTH = 756
cfg.HEIGH = 864
cfg.IMAGE_LOADER = "BNBNuv04Loader"
cfg.RNG_SEED= 9
cfg.DEBUG = False
cfg.CHANNELS=[2]
cfg.NCHANNELS = 1
cfg.SCALE = 100.0
cfg.IMIN = 0.5
cfg.IMAX = 10.0
cfg.HAS_RPN = True
cfg.SCALES = [756]
cfg.MAX_SIZE = 864
cfg.IOCFG = sys.argv[1]


import utils.root_handler
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from detect_output.det_out_factory import DetOutFactory

def detect(net, image_name, det_output):
    """Detect object classes in an image using pre-computed object proposals."""

    im = cfg.RH.get_image(int(image_name))

    scores, boxes = im_detect(net, int(image_name), im=im)

    # Visualize detections for each class
    CONF_THRESH = 0.0 #using 0.0 means write out all detections
    NMS_THRESH = 0.3
    
    event_data = { 'image_name' : image_name, 'bboxes' : [] }
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        if len(inds) == 0:
            print "No detections on {}".format(image_name)
            continue
            
        for i in inds:
            bbox  = dets[i, :4]
            score = dets[i, -1]
            
            event_data['bboxes'].append( { 'cls' : cls, 'box' : bbox, 'prob' : score } )
            
    det_output.write_event(event_data)

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = sys.argv[2]

    caffemodel = sys.argv[3]
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()

    #Always set as zero. Use CUDA_VISIBLE_DEVICES=X; to specify device ID, not caffe
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\n\t>>> Loaded network {:s} <<< '.format(caffemodel)

    im_names=range(int(cfg.NEXAMPLES))
    
    det_factory = DetOutFactory()

    det_output = det_factory.get("DetOutAscii",{'outfname': sys.argv[4] + ".txt"})

    # det_output = det_factory.get("DetOutLarcv",{'outfname': 'aho.root',
    #                                             'iniom'   : cfg.RH.IOM,
    #                                             'copy_input':True})
    
    
    for im_name in im_names:
        sys.stdout.write('Detect for index {}\r'.format(im_name))
        sys.stdout.flush()
        detect(net, im_name ,det_output)

    det_output.close()
