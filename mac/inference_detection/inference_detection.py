import _init_paths
from fast_rcnn.config import cfg
import numpy as np
import pandas as pd
import sys
import caffe, os, sys

from larcv import larcv
larcv.load_pyutil

###load the config, it's not worth using pyYAML, loading config line by line
###do it this way for verbose, rather than execfile(...)
config_=None
with open(sys.argv[3]) as f: config_=f.read().split("\n")
for c in config_:
    if c!="": 
        print "exec(",c,")"
        exec(c)

import utils.root_handler
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

NUM_MAX=1000
def model_flist(prefix):

    if not prefix.startswith('/'): 
        prefix = '%s/%s' % (os.getcwd(),prefix)

    indir  = prefix[0:prefix.rfind('/')]
    prefix = prefix[prefix.rfind('/')+1:len(prefix)]

    flist = [x for x in os.listdir(indir) if x.startswith(prefix) and (x.endswith('.caffemodel') or x.endswith('.caffemodel.h5'))]

    fmap={}
    for f in flist:
        ftmp = f.replace(prefix,'')
        ftmp = ftmp.replace('.caffemodel.h5','')
        ftmp = ftmp.replace('.caffemodel','')
        if not ftmp.isdigit(): continue
        fmap[int(ftmp)] = '%s/%s' % (indir,f)
    return fmap

#
# Get iteration + weight file list
#

PREFIX = sys.argv[2]
fmap = model_flist(PREFIX)
iters = fmap.keys()
iters.sort()
for i in iters: print i,fmap[i]

#
# Construct net
#
caffe.set_mode_gpu()

NETCFG = sys.argv[1]

#
# Retrieve 
#
inference_fname = NETCFG.replace('.prototxt','.txt')
inference_fname = "inference_%s" % inference_fname
for iter_num in iters:
    res = {}
    if os.path.isfile(inference_fname):
        for l in [ x for x in open(inference_fname,'r').read().split('\n') if len(x.split())==2 and x.split()[0].isdigit() ]:
            i,p = l.split()
            res[int(i)] = float(p)
    print "C"
    if iter_num in res: continue

    model = fmap[iter_num]
    
    net = caffe.Net( NETCFG, model, caffe.TEST)

    num_events=cfg.NEXAMPLES
    print "num_events:",cfg.NEXAMPLES
    if NUM_MAX and num_events > NUM_MAX : num_events = NUM_MAX

    print
    print 'Total number of events:',num_events
    print
    event_data=[]
    for image_name in xrange(int(cfg.NEXAMPLES)):

        #image_name = ttree entry number
        
        sys.stdout.write('Iteration %-4d\r' % (image_name))
        sys.stdout.flush()
        
        """Detect object classes in an image using pre-computed object proposals."""

        im = cfg.RH.get_image(int(image_name))

        scores, boxes = im_detect(net, int(image_name), im=im)

        # Visualize detections for each class
        CONF_THRESH = 0.0 # using 0.0 means write out all detections
        NMS_THRESH = 0.3
    
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
            
            event_data.append( [ image_name, cls, score, bbox[0], bbox[1], bbox[2], bbox[3] ] )

        # event data holds what we want
    df=pd.DataFrame(event_data,columns=['entry','class','prob','x1','y1','x2','y2'])
    # get the top box probability
    df_top_bbox= df.ix[ df.groupby("entry",sort=False).agg(lambda x: np.argmax(x['prob'])).prob.values ]

    res[iter_num] = df_top_bbox.prob.describe()['mean']

    print 'Iteration %-4d ... Accuracy %g           ' % (iter_num, res[iter_num])

    iters=res.keys()
    iters.sort()
    fout = open(inference_fname,'w')
    for i in iters:
        fout.write('%d %g\n' % (i,res[i]))
    fout.close()



