# compute shear bispectrum
import numpy as np
import healpy as hp
import curvedsky
import pickle
import prjlib

import time


p, f, r = prjlib.analysis_init()

sL = r.bp[:2]
bn = p.bn
bp = r.bp
bst = 2
lmax = p.olmax
btype = ['equi','fold','sque','isos']

# norm
print('compute norm')
hbsp = {}
for b in btype:
    hbsp[b] = curvedsky.bispec.bispec_norm(bn,bp,b,bst,sL=sL)

bl = {} 
xl = {}
for y in p.ymaps:
    bl[y] = np.zeros((p.snum,4,bn))
    xl[y] = np.zeros((p.snum,4,bn))

# compute binned bispectrum
for i in range(p.snmin,p.snmax):

    print(i)
    t1 = time.time()

    # load tau
    tlm, clm = pickle.load(open(f.qtbh['TT'].alm[i],"rb"))
    mf,  mfc = pickle.load(open(f.qtbh['TT'].mfb[i],"rb"))
    tlm -= mf
    tlm = -tlm[:p.olmax+1,:p.olmax+1]

    for y in p.ymaps:
        # load yalm
        ylm, ylm1, ylm2 = pickle.load(open(f.y[y].yalm[i],"rb"))
        ylm = ylm[:p.olmax+1,:p.olmax+1]
        alm = np.array((tlm,ylm))


        for bi, b in enumerate(btype):
            bl[y][i,bi,:] = curvedsky.bispec.bispec_bin(bn,bp,lmax,ylm,b,bst,sL=sL)    * np.sqrt(4*np.pi)/hbsp[b]
            xl[y][i,bi,:] = curvedsky.bispec.xbispec_bin(bn,bp,lmax,2,alm,b,bst,sL=sL) * np.sqrt(4*np.pi)/hbsp[b]

    t2 = time.time()
    print(t2-t1)

# output
for y in p.ymaps:
    if p.snmax>=2:
        np.savetxt(f.bsptbh[y].mbsp,(np.concatenate((r.bc[None,:],np.mean(bl[y][1:,:,:],axis=0),np.std(bl[y][1:,:,:],axis=0)))).T,fmt='%.5e')
        np.savetxt(f.bsptbh[y].mxsp,(np.concatenate((r.bc[None,:],np.mean(xl[y][1:,:,:],axis=0),np.std(xl[y][1:,:,:],axis=0)))).T,fmt='%.5e')
    if p.snmin==0:
        np.savetxt(f.bsptbh[y].obsp,(np.concatenate((r.bc[None,:],bl[y][0,:,:],np.std(bl[y][1:,:,:],axis=0)))).T,fmt='%.5e')
        np.savetxt(f.bsptbh[y].oxsp,(np.concatenate((r.bc[None,:],xl[y][0,:,:],np.std(xl[y][1:,:,:],axis=0)))).T,fmt='%.5e')


