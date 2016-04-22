#!/usr/bin/python 
import numpy as np

import sys

import matplotlib
import matplotlib.pyplot as plt

from ROOT import larcv
larcv.load_pyutil


iom = larcv.IOManager(larcv.IOManager.kREAD)

iom.add_in_file("/stage/vgenty/out.root")
iom.set_verbosity(0)
iom.initialize()

entries = None
with open("/stage/vgenty/Singledevkit3/train_1.txt") as f:
    entries = f.read()

entries = [im for im in entries.split("\n") if im != "" ] 

for i,entry in enumerate(entries):
    entry = int(entry)
    
    
    #EXACT COPY OF ROOT HANDLER
    iom.read_entry(entry)

    ev_img = iom.get_data(larcv.kProductImage2D,"fake_color")

    im  = larcv.as_ndarray( ev_img.Image2DArray()[0] )
    s   = im.shape
    imm = np.zeros([ s[0], s[1], 3 ])

    img_v = ev_img.Image2DArray()
    
    assert img_v.size() == 3

    for j in xrange(3):
        imm[:,:,j]  = larcv.as_ndarray( img_v[j] )
        imm[:,:,j] = imm[:,:,j].T

    imm[ imm < 5 ]   = 0
    imm[ imm > 400 ] = 400              
    
    imm = imm[::-1,:,:]
    #imm = imm[::-1,:,:]
    
    imm = imm[:,:,2:]

    # annos = None
    # with open( "/stage/vgenty/Singledevkit3/Annotations/{}.txt".format(entry)) as f:
    #     annos = f.read()

    annoz = None
    with open( "/stage/vgenty/Singledevkit3/Annotations/{}.txt".format(entry)) as f:
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
    

    # annos = annos.split(" ");
    # annos = annos[1:]

    # a = []
    # for anno in annos:
    #     anno = anno.rstrip()
    #     a.append(float(anno))


    fig,ax = plt.subplots(figsize = (12,12))
    plt.imshow(imm[:,:,0])
    plt.axis('off')
    for b in a_v:
        ax.add_patch(plt.Rectangle( (b[0],b[1]),
                                    b[2]-b[0], 
                                    b[3]-b[1],
                                    fill=False,edgecolor='red',linewidth=2.5) )

    # ax.add_patch(plt.Rectangle( (a[0],a[1]),a[2]-a[0], a[3]-a[1],fill=False,edgecolor='blue',linewidth=2.5) )

    ax.set_title("ENTRY: {}".format(entry))
    plt.savefig("f_entry_{}.png".format(entry))
    print "entry: {}".format(entry)
    # print "that's fine: {}".format(a)

    if i == 25:
        break
