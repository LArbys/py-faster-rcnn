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

cfg.ROOTFILES   = ["/data/drinkingkazu/production/v03/hires_crop/data_extbnb/part01/hiresdiv_extbnb_0000_0009.root"]

# /stage/drinkingkazu/production/v03/hires_crop/mc_bnb/hirescrop_bnb.root"]

cfg.IMAGE2DPROD  = "tpc_hires_crop"
cfg.ROIPROD = "tpc_hires_crop"
cfg.HEIGHT= 576
cfg.WIDTH = 576
cfg.DEVKIT = "HiresFilterDevkit"
cfg.IMAGE_LOADER = "SinglepLoader"
cfg.RNG_SEED= 7
cfg.NCHANNELS = 1
cfg.IMIN = 50
cfg.IMAX = 400
cfg.PIXEL_MEANS = np.array([[[ 0 ]]])
cfg.UB_N_CLASSES = 5

cfg.TEST.SCALES    = [576]
cfg.TEST.MAX_SIZE  = 576

CLASSES = ('__background__',
           'Eminus','Gamms','Muminus','Piminus','Proton')

from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt

import caffe, os, sys, cv2
import argparse
from ROOT import larcv

larcv.load_pyutil

import numpy as np
import lib.utils.root_handler as rh

NETS = {'rpn_uboone': ('alex_5_singlep',
                       'rpn_uboone_alex_5_singlep__iter_100000.caffemodel') }


#def vis_detections(im, class_name, dets, image_name, thresh=0.5):
def vis_detections(im, detections, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    
    thres_dets = {}
    for class_name,dets in detections.iteritems():
        inds = np.where(dets[:,-1] >= thresh)[0]
        thres_dets[class_name] = (dets,inds)
    
    imm = np.zeros([im.shape[0], im.shape[1]] + [3])

    for j in xrange(3):
        imm[:,:,j] = im[:,:,0]

    imm = imm.astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(imm, aspect='equal')

    # annos = None
    # with open( "data/{}/valid/{}.txt".format(cfg.DEVKIT,image_name) ) as f:
    #     annos = f.read()
    
    # annos = annos.split(" ");
    # truth = str(annos[0])
    # annos = annos[1:]

    # a = []
    # for anno in annos:
    #     anno = anno.rstrip()
    #     a.append(float(anno))
        
    # ax.add_patch(
    #     plt.Rectangle( (a[0],a[1]),a[2]-a[0], a[3]-a[1],fill=False,edgecolor='blue',linewidth=3.5) )
    k = 0
    for class_name,dets_inds in thres_dets.iteritems():
        print type(dets_inds)
        dets = dets_inds[0]
        inds = dets_inds[1]
        if len(inds) == 0: continue
        k+=1
        for i in inds:
            bbox  = dets[i, :4]
            score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('Truth=={}   Detection =={} with '
                  'p({} | box) >= {:.1f}').format("cosmic",
                                                  class_name, 
                                                  class_name,
                                                  thresh),fontsize=14)
    if k == 0:
        return

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig("{}_{}_demo_cosmic.png".format(image_name,class_name),format="png")
    
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""


    im = rh.get_image(int(image_name))
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, int(image_name), im=im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.4
    detections = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detections[cls] = dets

    vis_detections(im, detections, image_name,thresh=CONF_THRESH)

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

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
    cfg.MODELS_DIR = '/home/vgenty/py-faster-rcnn/models/rpn_uboone'

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')

    caffemodel = os.path.join('/stage/vgenty/faster_rcnn_end2end/rpn_uboone_train_5',NETS[args.demo_net][1])
    
    
    #/home/vgenty/py-faster-rcnn/output/faster_rcnn_alt_opt/rpn_uboone_train_5
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
    
    im_names = [i for i in xrange(250)]

    # for i in xrange(65):
    #     rand = np.random.random_integers(0,19000)
    #     if rand not in im_names:
    #         im_names.append(rand)
  
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
