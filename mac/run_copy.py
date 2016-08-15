import os,sys
import numpy as np

DIR = sys.argv[1]

print ""
done=[f.split(".txt")[0][12:] for f in os.listdir(".") if f.endswith(".txt")]
fs = [os.path.join(DIR,f) for f in os.listdir(DIR) if f.endswith(".caffemodel") if f not in done if "resnet" in f]

import sys

from multiprocessing.dummy import Pool as ThreadPool

def work(d):
    os.system("python tools/inference_nu_data_copy.py io_nu_valid.cfg %s"%d)
    
# Make the Pool of workers
pool = ThreadPool(1)

results = pool.map(work, fs)
    
#close the pool and wait for the work to finish
pool.close()
pool.join()

                             
