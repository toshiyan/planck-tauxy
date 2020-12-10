
import numpy as np, os
from quicksub import *

def jobfile(tag,r0,r1):

    # set job file
    f_run = 'run.py'
    f_sub = 'tmp_job_sub_'+tag+'.sh'
    set_sbatch_params(f_sub,tag,mem='8G',t='1-00:00',email=True)
    add('source ~/.bashrc.ext',f_sub)
    add('py4so',f_sub)
    add('python '+f_run,f_sub)
    
    # submit
    os.system('sbatch '+f_sub)


jobfile('run_nilc_LmaskN18',r0,r1)


