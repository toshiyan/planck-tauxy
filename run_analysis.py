# Reconstruction using quadratic estimator
import numpy as np
import prjlib
import tools_cmb
import tools_qrec
import tools_y


def analysis_flow(run_cmb,run_qrec,qrun,run_y,ytypes=['milca','nilc'],mtypes=[0,1],yascale=1.,kwargs_ov={},kwargs_cmb={}):

    kwargs_qrec = {\
        'rlmin':100, \
        'rlmax':2048, \
        'nside':1024, \
        'n0min':1, \
        'n0max':int(kwargs_cmb['snmax']/2), \
        'mfmin':1, \
        'mfmax':kwargs_cmb['snmax'], \
        'rdmin':1, \
        'rdmax':kwargs_cmb['snmax'], \
        'qlist':['TT'], \
        'rd4sim':False \
    }

    if kwargs_cmb['dtype'] == 'dr2_nilc':
        eps = 1e-6
        itns = 1000
    else:
        eps = 1e-5
        itns = 100
        
    kwargs_cinv = {\
        'chn':1, \
        'nsides' : [1024], \
        'lmaxs'  : [2048], \
        'eps'    : [eps], \
        'itns'   : [itns] \
    }

    # Main calculation
    if run_cmb:
        tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov,kwargs_cinv)

    if run_qrec:
        tools_qrec.interface(qrun=qrun,run=run_qrec,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)

    if run_y:
        # loop over y map options
        for ytype in ytypes:
            for masktype in mtypes:
                kwargs_y = {\
                    'ytype':ytype,\
                    'ascale':yascale,\
                    'masktype':masktype,\
                    'tausig':kwargs_cmb['tausig']\
                }
                tools_y.interface(run_y,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec,kwargs_y=kwargs_y)



#run_cmb = ['ptsr','alm','aps']
#run_cmb = ['tausim','alm','aps']
#run_cmb = ['alm','aps']
run_cmb = []

qrun = ['norm','qrec','n0','mean','aps']
#qrun = ['norm','qrec','n0','mean','rdn0','aps']
#qrun = ['aps']

#run_qrec = ['len','tau','src','tbh','tBH']
run_qrec = ['tau','tbh','tBH']
#run_qrec = ['tbh','tBH']
#run_qrec = ['tBH']
#run_qrec = []

#run_y = ['yalm']
#run_y = ['yalm','tauxy','tbhxy','tBHxy']
run_y = ['tauxy','tbhxy','tBHxy']
#run_y = ['tbhxy','tBHxy']
#run_y = ['tauxy']
#run_y = []

kwargs_ov   = {\
    'overwrite':True, \
    #'overwrite':False, \
    'verbose':True \
}

kwargs_cmb  = {\
    'snmin':0, \
    'snmax':100, \
    #'dtype':'dr2_nilc', \
    #'dtype':'dr2_smica', \
    'dtype':'dr3_nosz', \
    #'wtype':'Lmask', \
    #'wtype':'G60Lmask', \
    'wtype':'LmaskDR3', \
    #'wtype':'LmaskN18', \
    'ascale':1., \
    'lmax':2048, \
    'fltr':'cinv', \
    #'fltr':'none', \
    'tausig': False\
}


analysis_flow(run_cmb,run_qrec,qrun,run_y,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb)

#for dtype, wtype in [('dr2_nilc','Lmask'),('dr2_smica','Lmask'),('dr3_nosz','LmaskDR3'),('dr2_smica','G60Lmask')]:
#    kwargs_cmb['dtype'] = dtype
#    kwargs_cmb['wtype'] = wtype
#    analysis_flow(run_cmb,run_qrec,qrun,run_y,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb)

