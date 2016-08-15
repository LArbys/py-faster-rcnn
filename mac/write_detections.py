#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
import numpy as np
import sys
import caffe, os, sys, cv2

from larcv import larcv
larcv.load_pyutil


CLASSES = ('__background__', 2)

cfg.PIXEL_MEANS =  [[[ 0.0 ]]]
cfg.IMAGE2DPROD = "tpc"
cfg.ROIPROD = "tpc"
cfg.WIDTH = 756
cfg.HEIGH = 864
cfg.IMAGE_LOADER = "BNBNuv04Loader"
cfg.RNG_SEED= 9
cfg.DEBUG = False
cfg.CHANNELS=[2]
cfg.NCHANNELS = 1
cfg.IMIN = 0.5
cfg.IMAX = 10.0
cfg.HAS_RPN = True
cfg.SCALES = [756]
cfg.MAX_SIZE = 864
cfg.IOCFG = "io_{}_valid.cfg".format(sys.argv[1])

import utils.root_handler
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer


NETS = { 'rpn_uboone': ('resnet6x6_nu_weights_nu',
                        '/data/vgenty/brett/updated_resnet/nu_weights/resnet6x6_nu_weights_nu_iter_10000.caffemodel') }

def detect(net, image_name, det_output):
    """Detect object classes in an image using pre-computed object proposals."""

    im = cfg.RH.get_image(int(image_name))

    scores, boxes = im_detect(net, int(image_name), im=im)

    # Visualize detections for each class
    CONF_THRESH = 0.0 #using 0.0 means write out all detections
    NMS_THRESH = 0.3
    
    event_data = { 'image_name' : image_name, 'bboxes' = [] }
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= thresh)[0]

        if len(inds) == 0:
            print "No detections on {}".format(image_name)
            continue
            
        for i in inds:
            bbox  = dets[i, :4]
            score = dets[i, -1]
            
            event_data['bboxes'].append( { 'cls' : cls, 'box' : bbox, 'prob' : score } )
            
    print "event_data is ",event_data
    det_output.write_detections(event_data)

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.MODELS_DIR = '/home/vgenty/segment/py-faster-rcnn/models/rpn_uboone'

    prototxt = os.path.join(cfg.MODELS_DIR,NETS['rpn_uboone'][0],'faster_rcnn_end2end', 'test.prototxt')

    caffemodel = os.path.join(NETS['rpn_uboone'][1])
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    #Always set as zero. Use CUDA_VISIBLE_DEVICES=X; to specify device ID, not caffe
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\n\t>>> Loaded network {:s} <<< '.format(caffemodel)

    im_names=range(10000)
    
    
    det_factory = DetOutFactory()

    det_output = det_factory.get("DetOutLarcv",{'outfname'= 'aho.root',
                                                'iniom'   = cfg.RH.IOM,
                                                'copy_input'=True})
    

        
    for im_name in im_names:
        sys.stdout.write('Detect for index {}\r'.format(im_name))
        sys.stdout.flush()
        detect(net, im_name ,det_output)

    det_output.close()
