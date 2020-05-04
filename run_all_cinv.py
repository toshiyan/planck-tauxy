# Reconstruction using quadratic estimator
import numpy as np
import prjlib
import tools_cmb
import tools_qrec
import tools_y

#run_cmb  = ['ptsr','alm','aps']
#run_cmb  = ['tausim','alm','aps']
#run_cmb  = ['alm','aps']
run_cmb  = ['aps']
#run_cmb = []

qrun = ['norm','qrec','n0','mean','aps']
#qrun = ['norm','qrec','mean','aps']
#qrun = ['qrec']
#qrun = ['n0','mean','aps']

run_qrec = ['len','tau','src','tbh','tBH']
#run_qrec = ['len','src','tbh','tBH']
#run_qrec = ['tau']
#run_qrec = []

#run_y = ['yalm','tauxy','tbhxy','tBHxy']
#run_y = ['yalm']
run_y = ['tauxy','tbhxy','tBHxy']
#run_y = []

kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_cmb  = {\
    'snmin':1, \
    'snmax':100, \
    'dtype':'dr2_smica', \
    'wtype':'Lmask', \
    'ascale':1., \
    'lmax':2048, \
    'fltr':'cinv', \
    'tausig': True\
}

kwargs_qrec = {\
    'rlmin':100, \
    'rlmax':2048, \
    'nside':1024, \
    'n0min':1, \
    'n0max':int(kwargs_cmb['snmax']/2), \
    'mfmin':1, \
    'mfmax':kwargs_cmb['snmax'], \
    'rdmin':1, \
    'rdmax':kwargs_cmb['snmax'] \
}

kwargs_y = {\
    'ytype':'milca',\
    'ascale':1.,\
    'masktype':0,\
    'tausig':kwargs_cmb['tausig']\
}

kwargs_cinv = {\
    'chn':1, \
    'nsides' : [1024], \
    'lmaxs'  : [2048], \
    'eps'    : [1e-5], \
    'itns'   : [100] \
}

# Main calculation
if run_cmb:
    tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov,kwargs_cinv)

if run_qrec:
    tools_qrec.interface(qrun=qrun,run=run_qrec,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)

if run_y:
    tools_y.interface(run_y,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec,kwargs_y=kwargs_y)


