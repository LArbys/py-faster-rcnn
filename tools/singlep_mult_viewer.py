#!/usr/bin/python 
import numpy as np

import sys

import matplotlib
import matplotlib.pyplot as plt

from ROOT import larcv
larcv.load_pyutil

import _init_paths
from fast_rcnn.config import cfg

entries = None

with open("/stage/vgenty/Singledevkit6/train_4.txt") as f:
    entries = f.read()

entries = [im for im in entries.split("\n") if im != "" ] 

cfg.IMAGE_LOADER = "LarbysDetectLoader"
cfg.ROOTFILES   = ["/stage/vgenty/train_fakecolor_0.root"]
cfg.IMAGE2DPROD  = "fake_color"

import lib.utils.root_handler as rh

for i,entry in enumerate(entries):
    entry = int(entry)    
    
    imm = rh.get_image(entry)
    
    annoz = None
    with open( "/stage/vgenty/Singledevkit6/Annotations/{}.txt".format(entry)) as f:
        annoz = f.read()

    annos_v = annoz.split("\n")
    
    a_v = []
    for annoz in annos_v:
        if annoz == '': continue
        annoz = annoz.split(" ");
        annoz = annoz[1:]

        aa = []
        for anno in annoz:
            anno = anno.rstrip()
            aa.append(float(anno))
        a_v.append(aa)

    fig,ax = plt.subplots(figsize = (12,12))

    imm = imm.astype(np.uint8)
    plt.imshow(imm)

    plt.axis('off')
    for b in a_v:
        ax.add_patch(plt.Rectangle( (b[0],b[1]),
                                    b[2]-b[0], 
                                    b[3]-b[1],
                                    fill=False,edgecolor='red',linewidth=2.5) )


    ax.set_title("ENTRY: {}".format(entry))
    plt.savefig("f_entry_{}.png".format(entry))
    print "entry: {}".format(entry)
