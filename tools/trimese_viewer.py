import numpy as np
import sys

import lmdb
import caffe
from caffe.proto import caffe_pb2 as cpb
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

lmdb_env = lmdb.open( "/stage/vgenty/Singledevkit2/ccqe_supported_images_train.db" )
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

args = sys.argv[1:]
for image_name in args:
    datum = cpb.Datum()
    im = lmdb_cursor.get(image_name)
    datum.ParseFromString(im)
    im = caffe.io.datum_to_array(datum)
    im = np.transpose(im, (1,2,0))

    fig,ax = plt.subplots(figsize = (12,12))
    
    a = im[:,:,0:3].copy()
    a[:,:,0] = im[:,:,2]
    a[:,:,1] = im[:,:,6]
    a[:,:,2] = im[:,:,10]
    
    plt.imshow(a)
    plt.axis('off')

    annos = None
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
        

    ax.set_title("Union box: {}".format(image_name))
    plt.show()

