import numpy as np, prjlib, tools_qrec
kwargs_ov = {'overwrite':True,'verbose':True}
kwargs_cmb = {'snmin':30,'snmax':39,'dtype':'dr3_nosz','wtype':'LmaskDR3','ascale':1.,'lmax':2048,'fltr':'cinv','tausig':False}
kwargs_qrec = {'rlmin':100,'rlmax':2048,'nside':1024,'n0min':1,'n0max':50,'rdmin':1,'rdmax':100,'qlist':['TT'],'rd4sim':True}
tools_qrec.interface(qrun=['rdn0'],run=['tBH'],kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)
