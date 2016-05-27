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
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import lmdb
import caffe
from caffe.proto import caffe_pb2 as cpb


lmdb_env = lmdb.open( "/stage/vgenty/Singledevkit2/ccqe_supported_images_train.db" )
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

CLASSES = ('__background__',
           'neutrino')

NETS = {'rpn_uboone': ('trimese_2',
                       'rpn_uboone_trimese_2__iter_11850.caffemodel') }


def vis_detections(im, class_name, dets, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    
    print "after shape: {}".format(im.shape)

    fig, ax = plt.subplots(figsize=(12, 12))

    a = im[:,:,0:3].copy()
    a[:,:,0] = im[:,:,2]
    a[:,:,1] = im[:,:,6]
    a[:,:,2] = im[:,:,10]
    
    ax.imshow(a,aspect='equal')

    with open( "/stage/vgenty/Singledevkit2/Annotations/{}.txt".format(image_name)) as f:
        annos = f.read()
    
    annos = annos.split("\n");
    annos = [a for a in annos if a != '']
    
    for anno in annos:
        anno = anno.split(" ")
        a = anno[1:]
        a = [int(b) for b in a]
        ax.add_patch(
            plt.Rectangle( (a[0],a[1]),a[2]-a[0], a[3]-a[1],fill=False,edgecolor='blue',linewidth=3.5) )

    for i in inds:
        bbox = dets[i, :4]
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

    ax.set_title(('{} : {} detections with '
                  'p({} | box) >= {:.1f}').format(image_name,class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')


    plt.tight_layout()
    plt.draw()
    plt.savefig('det_%s_%s.png'%(image_name,class_name), format='png', dpi=100)

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image

    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im = cv2.imread(im_file)
    
    datum = cpb.Datum()
    im = lmdb_cursor.get(image_name)
    datum.ParseFromString(im)
    im = caffe.io.datum_to_array(datum)
    im = np.transpose(im, (1,2,0))
    
    # a = im[:, :, 0:3].copy()
    # a[:,:,0] = im[:,:,2]
    # a[:,:,1] = im[:,:,6]
    # a[:,:,2] = im[:,:,10]
    # im = a.copy()
    # detect all object classes and regress object bounds

    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    

    # Visualize detections for each class
    CONF_THRESH = 0.2
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
       
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        print "dets {}".format(dets)
        vis_detections(im, cls, dets, image_name,thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=2, type=int)
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
    
    cfg.MODELS_DIR = '/home/vgenty/py-faster-rcnn-lmdb/models/rpn_uboone'

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                         'faster_rcnn_alt_opt', 'fast_rcnn_test.pt')

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')

    caffemodel = os.path.join(cfg.DATA_DIR, '/home/vgenty/py-faster-rcnn-lmdb/output/faster_rcnn_end2end/rpn_uboone_train_1/',
                            NETS[args.demo_net][1])

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

    im_names = None

    with open("/stage/vgenty/Singledevkit2/train_1.txt") as f:
        im_names = f.read()

    im_names = [im for im in im_names.split("\n") if im != ""]
    #print im_names

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    #plt.show()
