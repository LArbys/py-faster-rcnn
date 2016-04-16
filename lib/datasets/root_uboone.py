# faster rcnn for uboone

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
#from root_uboone_eval import root_uboone_eval

from ROOT import larcv

from fast_rcnn.config import cfg

from utils.root_handler as rh

class root_uboone(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'root_uboone_' + image_set)
        
        self._image_set   = image_set
        self._devkit_path = "/stage/vgenty/ROOTDevKit/"
        
        self._data_path = os.path.join(self._devkit_path,"data")

        self._classes = tuple( ['__background__'] + cfg.UB_CLASSES )
        print "'\033[94m' \t >> Loaded UB classes {} '\033[0m'".format(self._classes)

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4()) #salt?

        # UBOONE specific config options
        self.config = {'use_salt'    : True,
                       'rpn_file'    : None,
                       'min_size'    : 2} # minimum box size
        
        assert os.path.exists(self._devkit_path), \
                'root_uboone path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        I suppose this is just _tree_entries
        """
        image_index = [ i for i in xrange( rh.get_n_entries() ) ]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where Superadevikit is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'Superadevkit')

    def gt_roidb(self): # can this become ROOT ?
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_uboone_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb    = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

        
    def rpn_roidb(self):
    
        if self._image_set != 'test':
            gt_roidb  = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb     = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb
        
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_uboone_annotation(self, index):
        """
        Load image and bounding boxes info from ROOT file in the UBOONE
        format.
        """
        # ask iomanager for this image's ROI (single plane for now be careful...)
        roidata  = get_roi_data(entry)
        num_objs = len(roidata)
        
        boxes      = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        
        # "Seg" area for uboone is just the box area -- what is this?
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes
        for ix, obj in enumerate(roidata):

            cls = self._class_to_ind[ int( obj['data'] ) ]

            x1 = float(obj['x1'])
            y1 = float(obj['y1'])
            x2 = float(obj['x2'])
            y2 = float(obj['y2'])
        
            boxes[ix, :]    = [x1, y1, x2, y2]

            gt_classes[ix]  = cls
            
            overlaps[ix, cls] = 1.0

            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes'       : boxes,
                'gt_classes'  : gt_classes,
                'gt_overlaps' : overlaps,
                'flipped'     : False,
                'seg_areas'   : seg_areas}

if __name__ == '__main__':
    raise RuntimeError("Factory must create me")
