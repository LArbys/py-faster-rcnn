#!/usr/bin/python

import os,sys

if len(sys.argv) != 3:
    print '\033[91m' + "\n\t Usage: alt.py [name] [numclasses]\n" + '\033[0m'
    sys.exit(1)

name, numclasses = sys.argv[1:]
print '\033[92m' + "\n\t generating google frcnn {} with {} of classes\n".format(name,numclasses) + '\033[0m'

numclasses = int(numclasses)
numclasses += 1

os.chdir("gen")

##########1
solver1 = None
with open("stage1_rpn_solver60k80k_google.pt",'r') as f:
    solver1 = f.read()

train1 = None
with open("stage1_rpn_train_google.pt",'r') as f:
    train1 = f.read()

############2
solver2 = None
with open("stage1_fast_rcnn_solver30k40k_google.pt",'r') as f:
    solver2 = f.read()

train2 = None
with open("stage1_fast_rcnn_train_google.pt",'r') as f:
    train2 = f.read()

##########3
solver3 = None
with open("stage2_rpn_solver60k80k_google.pt",'r') as f:
    solver3 = f.read()

train3 = None
with open("stage2_rpn_train_google.pt",'r') as f:
    train3 = f.read()


##########4
solver4 = None
with open("stage2_fast_rcnn_solver30k40k_google.pt",'r') as f:
    solver4 = f.read()

train4 = None
with open("stage2_fast_rcnn_train_google.pt",'r') as f:
    train4 = f.read()

#########Tests
test1 = None
with open("rpn_test_google.pt",'r') as f:
    test1 = f.read()

test2 = None
with open("fast_rcnn_test_google.pt",'r') as f:
    test2 = f.read()

to_replace = { 
    'stage1_rpn_solver60k80k' : solver1,
    'stage1_rpn_train'  : train1,

    'stage1_fast_rcnn_solver30k40k' : solver2,
    'stage1_fast_rcnn_train'  : train2,

    'stage2_rpn_solver60k80k' : solver3,
    'stage2_rpn_train'  : train3,

    'stage2_fast_rcnn_solver30k40k' : solver4,
    'stage2_fast_rcnn_train'  : train4,

    'fast_rcnn_test'   : test2,
    'rpn_test'   : test1,
}

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
os.mkdir(os.path.join(name,'faster_rcnn_alt_opt'))

#make write out etc
os.chdir(os.path.join(name,'faster_rcnn_alt_opt'))

for f in to_replace:
    ff = to_replace[f];
    a = open(f + ".pt","w+")
    a.write(ff)
    a.close()

print "\n"

print '\033[94m' + "\t !!!DONE!!!" + '\033[0m'
