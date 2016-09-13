# faster rcnn for uboone

import os,sys
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid

from fast_rcnn.config import cfg

import utils.root_handler

from larcv import larcv
larcv.load_pyutil

class rpn_uboone(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'rpn_uboone_' + image_set)
    
        self._image_set = image_set

        
        self._classes = tuple( ['__background__'] + cfg.UB_CLASSES )

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        # self._comp_id = 'comp4'
        
        # UBOONE specific config options
        self.config = { 'use_salt'    : True,
                        'rpn_file'    : None,
                        'min_size'    : 2 } # minimum box size
        
        # assert os.path.exists(self._devkit_path), \
        #         'rpn_uboone path does not exist: {}'.format(self._devkit_path)
        # assert os.path.exists(self._data_path), \
        #         'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        return index

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # image_set_file = os.path.join(self._data_path,self._image_set + '.txt')
        # assert os.path.exists(image_set_file), \
        #     'Path does not exist: {}'.format(image_set_file)
        # with open(image_set_file) as f:
        #     image_index = [x.strip() for x in f.readlines()]
        #     return image_index
        
        print "Returning just %d ints for now" %cfg.NEXAMPLES
        return np.arange(0,cfg.NEXAMPLES)


    def _get_default_path(self):
        """
        Return the default path where Singlesdevikit is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, cfg.DEVKIT)

    def gt_roidb(self):
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
        print "\t Hey vic I was called, don't delete me! __selective_search__"
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
    
        print "\t Hey vic I was called, don't delete me! __rpn_roidb__"

        if self._image_set != 'test':
            gt_roidb  = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb     = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb
        
    #### NOT SURE IF THIS IS CALLED
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    ### NOT SURE IF THIS IS CALLED
    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)

        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []

        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)]
            keep  = ds_utils.unique_boxes(boxes)
            
            boxes = boxes[keep, :]
            keep  = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            
            boxes = boxes[keep, :]
            
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_uboone_annotation(self, index):
        """
        Load image and bounding boxes info from ROOT file into mem
        format.
        """
        sys.stdout.write('{} imported\r'.format(index))
        sys.stdout.flush()

        num_objs=0
        if cfg.RH is None:
            raise Exception("cfg.rh is None")

        cfg.RH.IOM.read_entry(index)
        
        xy = {}
        
        ev_image = cfg.RH.IOM.get_data(larcv.kProductImage2D,cfg.IMAGE2DPROD)
        ev_roi   = cfg.RH.IOM.get_data(larcv.kProductROI,cfg.ROIPROD)
    
        image_v = ev_image.Image2DArray()
        images = [ image_v[k] for k in cfg.CHANNELS ]
            
        roi_v = ev_roi.ROIArray()
        
        # for each ROI
        for ix,roi in enumerate(roi_v):

            roitype = larcv.PDG2ROIType(roi.PdgCode())
            #is it one of the UB classes?
            
            if roitype not in self._classes:
                if cfg.DEBUG:
                    print roitype, "Not in self._classes: ",self._classes
                continue
                
            if roitype not in xy.keys():
                xy[roitype]=[]            

            for ix,channel in enumerate(cfg.CHANNELS):
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
                xy_conv= (imrows-y2,x1,imrows-y1,x2)
                if xy_conv in xy[roitype] : continue
                xy[roitype].append(xy_conv)
                num_objs+=1;
        
                
        # at this point we could have duplicates, why? b/c larbys jesus says so

        if cfg.DEBUG:
            print xy
        
        boxes      = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        
        # "Seg" area for uboone is just the box area -- what is this?
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame -- what dataframe?
        iy=-1
        for roitype in xy:
            for roi in xy[roitype]:
                iy+=1
                cls = self._class_to_ind[roitype]
            
                x1 = float(roi[0])
                y1 = float(roi[1])
                x2 = float(roi[2])
                y2 = float(roi[3])
        
                boxes[iy, :] = [x1, y1, x2, y2]

                gt_classes[iy]  = cls
            
                overlaps[iy, cls] = 1.0

                seg_areas[iy] = (x2 - x1 + 1) * (y2 - y1 + 1)
                
        overlaps = scipy.sparse.csr_matrix(overlaps)


        return {'boxes'       : boxes,
                'gt_classes'  : gt_classes,
                'gt_overlaps' : overlaps,
                'flipped'     : False,
                'seg_areas'   : seg_areas}

    def _get_comp_id(self):
        raise Exception("Unsupported function call")
        # comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
        #     else self._comp_id)

        # return comp_id


    def _get_ub_results_file_template(self):
        raise Exception("Unsupported function call")
        # filename = self._get_comp_id() + '_' + self._name  + '_det_' + self._image_set + '_{:s}.txt'
        # path = os.path.join(
        #     self._devkit_path,
        #     'results',
        #     'Main',
        #     filename)
        # return path

    def _write_ub_results_file(self, all_boxes):
        raise Exception("Unsupported function call")
        # for cls_ind, cls in enumerate(self.classes):
        #     if cls == '__background__':
        #         continue
        #     print 'Writing {} ub results file'.format(cls)
        #     filename = self._get_ub_results_file_template().format(cls)
        #     with open(filename, 'wt') as f:
        #         for im_ind, index in enumerate(self.image_index):
        #             dets = all_boxes[cls_ind][im_ind]
        #             if dets == []:
        #                 continue
        #             # the ub expects 0-based indices
        #             for k in xrange(dets.shape[0]):
        #                 f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {}\n'.
        #                         format(index, dets[k, -1],
        #                                dets[k, 0], dets[k, 1],
        #                                dets[k, 2], dets[k, 3],
        #                                cls_ind))


    def _do_python_eval(self, output_dir = 'output'):
        raise Exception("Unsupported function call")
        # annopath = os.path.join(
        #     self._devkit_path,
        #     'Annotations',
        #     '{:s}.txt')
        # imagesetfile = os.path.join(
        #     self._devkit_path,
        #     'ImageSets',
        #     'Main',
        #     self._image_set + '.txt')
        # cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        # aps = []

        # if not os.path.isdir(output_dir):
        #     os.mkdir(output_dir)
        # for i, cls in enumerate(self._classes):
        #     if cls == '__background__':
        #         continue
        #     filename = self._get_ub_results_file_template().format(cls)
        #     rec, prec, ap = rpn_uboone_eval(
        #         filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.25)
        #     aps += [ap]
        #     print('AP   for {} = {:.4f}'.format(cls, ap))
        #     with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
        #         cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        # print('Mean AP = {:.4f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('\tComplete')

    def evaluate_detections(self, all_boxes, output_dir):
        raise Exception("Unsupported function call")
        # self._write_ub_results_file(all_boxes)
        # self._do_python_eval(output_dir)
        
    def _get_widths(self):
        return [ int(cfg.WIDTH) for i in xrange(self.num_images) ]


if __name__ == '__main__':
    from datasets.rpn_uboone import rpn_uboone

    d = rpn_uboone('trainval') #no choice yet, must trainval
    res = d.roidb
    
    from IPython import embed; embed()
