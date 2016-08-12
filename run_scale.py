import os

# SCALES=['0.000',
#         '0.0005',
#         '-0.0005',
#         '0.001',
#         '-0.001',
#         '0.00025',
#         '-0.00025',
#         '0.0015',
#         '-0.0015']
SCALES=['0.002','-0.002']
GPU=0
for SCALE in SCALES:
    os.system("python tools/demo_nu_v04_scaling.py --net rpn_uboone --gpu {} --scale {}&".format(GPU,SCALE))
    GPU+=1
    GPU=GPU%4
    
