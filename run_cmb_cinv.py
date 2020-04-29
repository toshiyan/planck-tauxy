# Reconstruction using quadratic estimator
import numpy as np
import prjlib
import tools_cmb
import tools_qrec


#run_cmb  = ['alm','aps']
run_cmb = []

#run_qrec = ['norm','qrec','n0','mean','aps']
run_qrec = ['rdn0']
#run_qrec = []
#qrec = ['len','tau','tbh']
#qrec = ['len','tau']
qrec = ['tau']

kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_cmb  = {\
    'snmin':0, \
    'snmax':100, \
    'dtype':'dr2_smica', \
    'wtype':'Lmask', \
    'ascale':1., \
    'lmax':2048, \
    'fltr':'cinv' \
}

kwargs_qrec = {\
    'rlmin':100, \
    'rlmax':2048, \
    'nside':2048, \
    'n0min':1, \
    'n0max':50, \
    'mfmin':1, \
    'mfmax':100, \
    'rdmin':1, \
    'rdmax':100 \
}

kwargs_cinv = {\
    'chn':1, \
    'nsides' : [2048], \
    'lmaxs'  : [2048], \
    'eps'    : [1e-5], \
    'itns'   : [500] \
}

# Main calculation
if run_cmb:
    tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov,kwargs_cinv)

if run_qrec:
    tools_qrec.interface(run_qrec,qrec,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)


