import numpy as np
import sys

import lmdb
import caffe
from caffe.proto import caffe_pb2 as cpb
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

lmdb_env = lmdb.open( "/stage/vgenty/Singledevkit2/ccqe_supported_images_test.db" )
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

images_names = None
with open("/home/vgenty/files.idx") as f:
    image_names = f.read()

image_names = [im for im in image_names.split("\n") if im != "" ] 

for image_name in image_names:

    datum = cpb.Datum()
    print image_name
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
    try:
        with open( "/stage/vgenty/Singledevkit2/valid/{}.txt".format(image_name)) as f:
            annos = f.read()
    except IOError:
        continue
    
    annos = annos.split(" ");
    annos = annos[1:]
    
    a = []
    for anno in annos:
        anno = anno.rstrip()
        a.append(int(anno))
        
    ax.add_patch(
    plt.Rectangle( (a[0],a[1]),a[2]-a[0], a[3]-a[1],fill=False,edgecolor='blue',linewidth=3.5) )
    ax.set_title("Union box: {}".format(image_name))
    plt.savefig("union_{}.png".format(image_name))

