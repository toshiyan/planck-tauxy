
import numpy as np, os
from quicksub import *

def create_runfile(f,r0,r1,dtype='dr2_nilc',wtype='Lmask'):
    
    add('import numpy as np, prjlib, tools_cmb',f,ini=True)
    add("kwargs_ov = {'overwrite':True,'verbose':True}",f)
    add("kwargs_cmb = {'snmin':"+str(r0)+",'snmax':"+str(r1)+",'dtype':'"+dtype+"','wtype':'"+wtype+"','ascale':1.,'lmax':2048,'fltr':'cinv','tausig':False}",f)
    add("kwargs_cinv = {'chn':1,'nsides':[1024],'lmaxs':[2048],'eps':[5e-7],'itns':[5000]}",f)
    add("tools_cmb.interface(['alm','aps'],kwargs_cmb,kwargs_ov,kwargs_cinv)",f)


def jobfile(tag,r0,r1,**kwargs):

    #set run file
    f_run = 'tmp_job_run_'+tag+'.py'
    create_runfile(f_run,r0,r1,**kwargs)

    # set job file
    f_sub = 'tmp_job_sub_'+tag+'.sh'
    set_sbatch_params(f_sub,tag,mem='16G',t='0-10:00',email=True)
    add('source ~/.bashrc.ext',f_sub)
    add('py4so',f_sub)
    add('python '+f_run,f_sub)
    
    # submit
    os.system('sbatch '+f_sub)
    #os.system('sh '+f_sub)
    #os.system('rm -rf '+f_run+' '+f_sub)


rlz0 = np.arange(0,120,10)
rlz1 = rlz0[1:]-1
rlz1[-1] = 100

for r0, r1 in zip(rlz0,rlz1):
    print(r0,r1)
    jobfile('cinv_rlz_'+str(r0)+'-'+str(r1),r0,r1)

