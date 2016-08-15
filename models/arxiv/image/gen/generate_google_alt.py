#!/usr/bin/python

import os,sys
from collections import OrderedDict

if len(sys.argv) != 4:
    print '\033[91m' + "\n\t Usage: alt.py [name] [numclasses] [4a/4b/4c]\n" + '\033[0m'
    sys.exit(1)

name, numclasses, inc = sys.argv[1:]
os.system( "rm -rf {}".format(name) )

print '\033[92m' + "\n\t generating google alt opt {} with {} of classes\n".format(name,numclasses) + '\033[0m'

numclasses = int(numclasses)
numclasses += 1
bboxpred = str(numclasses*4)
numclasses = str(numclasses)


os.chdir("gen/google")

def get_data(infile):
    data= None
    with open(infile,'r') as f:
        data = f.read()
    return data

inceptions = OrderedDict()
inceptions['4a'] = get_data('__inception_4a.pt')
inceptions['4b'] = get_data('__inception_4b.pt')
inceptions['4c'] = get_data('__inception_4c.pt')

inputs = OrderedDict()
inputs['4a'] = ''
inputs['4b'] = 'inception_4a/output'
inputs['4c'] = 'inception_4b/output'

outputs = OrderedDict()
outputs['4a'] = 'inception_4a/output'
outputs['4b'] = 'inception_4b/output'
outputs['4c'] = 'inception_4c/output'

last = OrderedDict()
last['4a'] = 'inception_4c/output'
last['4b'] = 'inception_4c/output'
last['4c'] = 'RPNORPOOL'


def before_ic(ic):
    before = ''
    for k, v in inceptions.items():
        v = v.replace("INPUT",inputs[k])
        before += v
        
        if k == ic:
            return before

def after_ic(ic):
    after = ''
    c = 0
    for k, v in inceptions.items():
        if inceptions.keys().index(k) <= inceptions.keys().index(ic):
            continue
        if c > 0:
            v = v.replace("INPUT",inputs[k])
        c += 1
        after += v
        
    return after

###TRAINERS
#by hand
trainers = {'rpn'  : { 'stage1' : 'stage1_rpn_train.pt'      , 'stage2' : 'stage2_rpn_train.pt'},
            'frcnn': { 'stage1' : 'stage1_fast_rcnn_train.pt', 'stage2' : 'stage2_fast_rcnn_train.pt'}}

otrainers = {}

for t in trainers:
    th   = ''
    if t == 'rpn':
        #Header
        th  = get_data('__rpn_header.pt')

        #Google top
        th += get_data('__googlenet_top.pt')

        #Upto
        th += before_ic(inc)

        #RPN layers
        th += get_data('__rpn_layers.pt')
        th  = th.replace("INPUT",outputs[inc])

        #Silences
        th += after_ic(inc)
        th = th.replace("INPUT",'dummy_roi_pool_conv5')
        th += get_data('__googlenet_bottom.pt')
        th = th.replace("INPUT",last[inc]) #LAST == 4c for now
        th = th.replace("RPNORPOOL",'dummy_roi_pool_conv5')

        th += get_data("__rpn_silence.pt")
        
        #replacements
        th = th.replace("NAME",name)
        th = th.replace("SD1",'1')
        th = th.replace("SD2",'512')
        th = th.replace("SD3",'37')
        th = th.replace("SD4",'62')
            

    if t == 'frcnn':
        #Header
        th  = get_data('__frcnn_header.pt')

        #Google top
        th += get_data('__googlenet_top.pt')

        #Upto
        th += before_ic(inc)

        #POOLERS
        th += get_data('__frcnn_pooler.pt')
        th  = th.replace("INPUT",outputs[inc])
        
        # After
        th += after_ic(inc)
        th  = th.replace("INPUT",'pool5')

        # Rest of google
        th += get_data('__googlenet_bottom.pt')
        th = th.replace("INPUT",last[inc]) #LAST == 4c for now
        th = th.replace("RPNORPOOL",'pool5')
        
        #RPN layers
        th += get_data('__frcnn_layers.pt')
        th  = th.replace("INPUT",outputs[inc])

        
        #replacements
        th = th.replace("NAME",name)
        th = th.replace("NUMCLASSES",numclasses)
        th = th.replace("BBOXPRED",bboxpred)
        th = th.replace("POOLEDH",str(14))
        th = th.replace("POOLEDW",str(14))

        
    for stage in trainers[t]:
        a = th #copy the string
        ofile = trainers[t][stage]

        if stage == 'stage1':
            a = a.replace("WEIGHT_LR",str(1))
            a = a.replace("BIAS_LR"  ,str(2))

        if stage == 'stage2':
            a = a.replace("WEIGHT_LR",str(0))
            a = a.replace("BIAS_LR"  ,str(0))

        otrainers[ofile] = a
            
        
###SOLVERS
solvers  = {}
osolvers = {}

solvers['stage1_fast_rcnn_solver30k40k.pt'] = get_data('stage1_fast_rcnn_solver30k40k_google.pt')
solvers['stage2_fast_rcnn_solver30k40k.pt'] = get_data('stage2_fast_rcnn_solver30k40k_google.pt')

solvers['stage1_rpn_solver60k80k.pt'] = get_data('stage1_rpn_solver60k80k_google.pt')
solvers['stage2_rpn_solver60k80k.pt'] = get_data('stage2_rpn_solver60k80k_google.pt')

for s in solvers:
    ff = solvers[s];
    ff = ff.replace('NAME',name)
    osolvers[s] = ff 
    

###Tests
#by hand
tests = {'rpn'  : 'rpn_test.pt',
         'frcnn': 'fast_rcnn_test.pt' }
otest = {}


for t in tests:
    th   = ''
    if t == 'rpn':
        #Header
        th  = get_data('__test_header.pt')
        #Google top
        th += get_data('__googlenet_top.pt')
        #Upto
        th += before_ic(inc)
        #RPN test
        th += get_data('__rpn_test_layers.pt')
        th  = th.replace("INPUT",outputs[inc])
        #replacements
        th = th.replace("NAME",name)
        th = th.replace("WEIGHT_LR",str(1))
        th = th.replace("BIAS_LR"  ,str(2))
                
    if t == 'frcnn':
        #Header
        th  = get_data('__test_header.pt')

        #Google top
        th += get_data('__googlenet_top.pt')

        #Upto
        th += before_ic(inc)

        #FRCNN test
        th += get_data('__frcnn_test_layers.pt')
        th = th.replace("INPUT",outputs[inc])        

        #Upto
        th += after_ic(inc)
        th = th.replace("INPUT",'pool5')
        
        #Bottom
        th += get_data('__googlenet_bottom.pt')
        th = th.replace("INPUT",last[inc])
        th = th.replace("RPNORPOOL",'pool5')
        
        #FRCNN Bottom
        th += get_data('__frcnn_test_bottom.pt')
                
        #replacements
        th = th.replace("NAME",name)
        th = th.replace("WEIGHT_LR",str(1))
        th = th.replace("BIAS_LR"  ,str(2))
        th = th.replace("NUMCLASSES",numclasses)
        th = th.replace("BBOXPRED",bboxpred)
        th = th.replace("POOLEDH",str(14))
        th = th.replace("POOLEDW",str(14))
        
    otest[tests[t]] = th

# go back two directory
os.chdir('../..')

# make name directory
os.mkdir(name)
os.mkdir(os.path.join(name,'faster_rcnn_alt_opt'))

#make write out etc
os.chdir(os.path.join(name,'faster_rcnn_alt_opt'))


def writeout(output):
    for o in output:
        out = output[o]
        a = open(o,"w+")
        a.write(out)
        a.close()

writeout(osolvers)
writeout(otest)
writeout(otrainers)

print "\n"

print '\033[94m' + "\t !!!DONE!!!" + '\033[0m'
