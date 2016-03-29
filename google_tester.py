import lmdb
import os,sys
import numpy as np


caffe_home='/home/vgenty/caffe/'
sys.path.insert(0, caffe_home + 'python')
import caffe

caffe.set_mode_gpu()

n_classes = int(sys.argv[1])
model     = sys.argv[2]

model_file      = '/home/vgenty/caffe/models/ub_googlenet/train_{}.prototxt'.format(n_classes)
pretrained_file = model
validate_data   = '/stage/vgenty/ub/ub_valid_{}.db'.format(n_classes)


net = caffe.Net( model_file, pretrained_file, caffe.TEST )
lmdb_name = validate_data
lmdb_env = lmdb.open(lmdb_name, readonly=True)
lmdb_txn = lmdb_env.begin()

cursor = lmdb_txn.cursor()

batchsize = 1
nbatches  = int(n_classes*1000)
totevents = 0

f = open('google_{}_class_results.txt'.format(n_classes),'w+')
f.write("key,label,decision\n")
for ibatch in range(0,nbatches):
    keys = []
    for iimg in range(0,batchsize):
        cursor.next()
        (key,raw_datum) = cursor.item()
        keys.append(key)

    net.forward()
    labels  =  net.blobs["label"].data
    scores  =  net.blobs["loss3/classifier"].data
    softmax =  net.blobs["loss3/loss3"].data

    totevents += float( len(scores) )
    for label,score,key in zip(labels,scores,keys):
        ilabel = int(label)
        decision = np.argmax(score)
        f.write( "{},{},{}\n".format(key,label,decision) )

f.close()
print("TOTEVENTS: {}".format(totevents))
