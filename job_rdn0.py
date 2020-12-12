
import numpy as np, os
from quicksub import *

def create_runfile(f,r0,r1,dtype='dr2_smica',wtype='Lmask',fltr='cinv'):
    
    add('import numpy as np, prjlib, tools_qrec',f,ini=True)
    add("kwargs_ov = {'overwrite':True,'verbose':True}",f)
    add("kwargs_cmb = {'snmin':"+str(r0)+",'snmax':"+str(r1)+",'dtype':'"+dtype+"','wtype':'"+wtype+"','ascale':1.,'lmax':2048,'fltr':'"+fltr+"','tausig':False}",f)
    add("kwargs_qrec = {'rlmin':100,'rlmax':2048,'nside':1024,'n0min':1,'n0max':50,'rdmin':1,'rdmax':100,'qlist':['TT'],'rd4sim':True}",f)
    add("tools_qrec.interface(qrun=['rdn0'],run=['tBH'],kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)",f)
    #add("tools_qrec.interface(qrun=['rdn0'],run=['tau','tbh','tBH'],kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)",f)


def jobfile(tag,r0,r1,**kwargs):

    #set run file
    f_run = 'tmp_job_run_'+tag+'.py'
    create_runfile(f_run,r0,r1,**kwargs)

    # set job file
    f_sub = 'tmp_job_sub_'+tag+'.sh'
    set_sbatch_params(f_sub,tag,mem='16G',t='0-20:00',email=True)
    add('source ~/.bashrc.ext',f_sub)
    add('py4so',f_sub)
    add('python '+f_run,f_sub)
    
    # submit
    os.system('sbatch '+f_sub)
    #os.system('sh '+f_sub)
    #os.system('rm -rf '+f_run+' '+f_sub)


rlz0 = np.arange(0,120,10)
rlz0 = np.arange(10,120,10)
rlz1 = rlz0[1:]-1
rlz1[-1] = 100

for r0, r1 in zip(rlz0,rlz1):
    print(r0,r1)
    #jobfile('base_rlz_'+str(r0)+'-'+str(r1),r0,r1,dtype='dr2_smica',wtype='Lmask')
    #jobfile('G60L_rlz_'+str(r0)+'-'+str(r1),r0,r1,dtype='dr2_smica',wtype='G60Lmask')
    #jobfile('NILC_rlz_'+str(r0)+'-'+str(r1),r0,r1,dtype='dr2_nilc',wtype='Lmask')
    jobfile('NoSZ_rlz_'+str(r0)+'-'+str(r1),r0,r1,dtype='dr3_nosz',wtype='LmaskDR3')
    #jobfile('TN18_rlz_'+str(r0)+'-'+str(r1),r0,r1,dtype='dr2_nilc',wtype='LmaskN18',fltr='none')


