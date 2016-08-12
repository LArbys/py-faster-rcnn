# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.rpn_uboone import rpn_uboone
from fast_rcnn.config import cfg

import numpy as np

# Setup rpn_uboone
for split in ['train_'    + str(cfg.UB_N_CLASSES),\
              'val_'      + str(cfg.UB_N_CLASSES),\
              'trainval_' + str(cfg.UB_N_CLASSES),\
              'test_'     + str(cfg.UB_N_CLASSES)] :
    
    print "Current split is ",split
    name = 'rpn_uboone_{}'.format(split)
    __sets[name] = (lambda split=split :  rpn_uboone(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
