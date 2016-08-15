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
import sys,os
scale = str(sys.argv[-1])

PATH='/data/vgenty/overlays/'
FILE='overlay_0000_0037_'

#SCALES=['0.000',
#        '0.0005',
#        '-0.0005',
#        '0.001',
#        '-0.001',
#        '0.00025',
#        '-0.00025',
#        '0.0015',
#        '-0.0015']

cfg.ROOTFILES   = [os.path.join(PATH,FILE+scale+'.root')]

CLASSES = ('__background__','neutrino')

cfg.PIXEL_MEANS =  [[[ 0.0 ]]]
cfg.IMAGE2DPROD = "tpc"
cfg.ROIPROD = "tpc"

#cfg.HEIGHT= 756
#cfg.WIDTH = 864

cfg.WIDTH = 756
cfg.HEIGH = 864
cfg.DEVKIT = "NuDevKitv04"
cfg.IMAGE_LOADER = "BNBNuv04Loader"
cfg.RNG_SEED= 9
cfg.DEBUG = False
cfg.NCHANNELS = 1
cfg.IMIN = 0.5
cfg.IMAX = 10.0
cfg.HAS_RPN = True
cfg.SCALES = [756]
cfg.MAX_SIZE = 864

from fast_rcnn.test import im_detect, rh
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt

import caffe, os, sys, cv2
import argparse
from ROOT import larcv

larcv.load_pyutil

import numpy as np

NETS = {'rpn_uboone': ('alex_nu_v04',
                       'earlier/rpn_uboone_alex_nu_v04__iter_44000.caffemodel') }


def vis_detections(im, class_name, dets, image_name, thresh=0.5,scale=None):
    """Draw detected bounding boxes."""
    
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print "No detections on {}".format(image_name)
        return
              
    for i in inds:
        bbox  = dets[i, :4]
        score = dets[i, -1]
        out = open("/stage/vgenty/scaling_study/scaling_dets_{}.txt".format(scale),"a")
        out.write("{} {} {} {} {} {}\n".format(image_name,
                                               score,
                                               bbox[0],
                                               bbox[1],
                                               bbox[2],
                                               bbox[3]))

    out.close()
    
def demo(net, image_name,scale=scale):
    """Detect object classes in an image using pre-computed object proposals."""


    im = rh.get_image(int(image_name))
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, int(image_name), im=im)
    timer.toc()

    # Visualize detections for each class
    CONF_THRESH = 0.0 # 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print dets
        vis_detections(im, cls, dets, image_name,thresh=CONF_THRESH,scale=scale)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    parser.add_argument('--scale', dest='adc_scale', help='ADC shift value',
                        default='0.000',type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    cfg.MODELS_DIR = '/home/vgenty/py-faster-rcnn/models/rpn_uboone'

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')

    caffemodel = os.path.join('/stage/vgenty/faster_rcnn_end2end/rpn_uboone_train_1',NETS[args.demo_net][1])
    
    # /home/vgenty/py-faster-rcnn/output/faster_rcnn_alt_opt/rpn_uboone_train_5
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    valids = None

    import ROOT
    from ROOT import larcv
    iom = larcv.IOManager()   

    iom.add_in_file(cfg.ROOTFILES[0])
    iom.initialize()
    entries = int(iom.get_n_entries())

    im_names = [ int(v) for v in xrange(entries) ]
    
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name,scale=args.adc_scale)
