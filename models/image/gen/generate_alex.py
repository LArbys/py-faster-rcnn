#!/usr/bin/python

import os,sys

if len(sys.argv) != 3:
    print '\033[91m' + "\n\t Usage: generate_alex.py [name] [numclasses]\n" + '\033[0m'
    sys.exit(1)

name, numclasses = sys.argv[1:]
print '\033[92m' + "\n\t generating alex frcnn {} with {} of classes\n".format(name,numclasses) + '\033[0m'

numclasses = int(numclasses)
numclasses += 1

os.chdir("gen")

solver = None
with open("alex_solver.prototxt",'r') as f:
    solver = f.read()

train = None
with open("alex_train.prototxt",'r') as f:
    train = f.read()

test = None
with open("alex_test.prototxt",'r') as f:
    test = f.read()
    

to_replace = { 'solver' : solver,
               'train'  : train,
               'test'   : test  }

for f in to_replace:
    ff = to_replace[f];
    ff = ff.replace('NAME',name)
    ff = ff.replace('NUMCLASSES',str(numclasses))
    ff = ff.replace('BBOXPRED',str(numclasses*4))
    to_replace[f] = ff #is this needed?
    
# go back one directory
os.chdir('..')

# make name directory
os.mkdir(name)
os.mkdir(os.path.join(name,'faster_rcnn_end2end'))

#make write out etc
os.chdir(os.path.join(name,'faster_rcnn_end2end'))

for f in to_replace:
    ff = to_replace[f];
    a = open(f + ".prototxt","w+")
    a.write(ff)
    a.close()

print "\n"

print '\033[94m' + "\t !!!DONE!!!" + '\033[0m'
