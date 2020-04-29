# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle

# from cmblensplus/wrap
import curvedsky
import basic

# from cmblensplus/utils
import quad_func
import misctools
import cmb as CMB

# local
import prjlib


def ymap2yalm(cy,Wy,rlz,lmax,w2):

    nside = hp.pixelfunc.get_nside(Wy)
    l = np.linspace(0,lmax,lmax+1)

    for i in rlz:

        misctools.progress(i,rlz,text='Current progress',addtext='(ymap2alm)')
        yalm = {}
        clyy = {}

        for yn in [0,1,2]: #full, first, last
        
            if i==0: # real data
                bl = CMB.beam(10.,lmax)
                ymap = Wy * hp.fitsfunc.read_map(cy.fymap,field=yn)
                yalm[yn] = curvedsky.utils.hp_map2alm(nside,lmax,lmax,ymap) * bl[:,None]
                clyy[yn] = curvedsky.utils.alm2cl(lmax,yalm[yn])/w2
            else: # generate sim
                clyy[yn] = np.loadtxt(cy.fclyy,unpack=True)[yn-1]
                galm = curvedsky.utils.gauss1alm(lmax,clyy[yn][:lmax+1])
                ymap = Wy * curvedsky.utils.hp_alm2map(nside,lmax,lmax,galm)
                yalm[yn] = curvedsky.utils.hp_map2alm(nside,lmax,lmax,ymap)

        if i == 0:
            xlyy = curvedsky.utils.alm2cl(lmax,yalm[1],yalm[2])/w2
            np.savetxt(cy.fclyy,np.array((l,clyy[0],clyy[1],clyy[2],xlyy)).T)
        pickle.dump((yalm[0],yalm[1],yalm[2]),open(cy.fyalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def tauxy(cy,lmax,bn,rlz,qf,fx,w0,w1,**kwargs_ov):

    __, bc = basic.aps.binning(bn,[1,lmax],'')
    l = np.linspace(0,lmax,lmax+1)
    
    xl = np.zeros((len(rlz),3,lmax+1))
    xb = np.zeros((len(rlz),3,bn))

    for i in rlz:

        print(i)

        # load tau
        tlm = pickle.load(open(qf['TT'].alm[i],"rb"))[0][:lmax+1,:lmax+1]
        mf  = pickle.load(open(qf['TT'].mfb[i],"rb"))[0][:lmax+1,:lmax+1]
        tlm -= mf
        tlm = -tlm
            
        # load yalm
        ylm = {}
        ylm[0], ylm[1], ylm[2] = pickle.load(open(cy.fyalm[i],"rb"))

        # cross spectra
        for yn in range(3):
            xl[i,yn,:] = curvedsky.utils.alm2cl(lmax,tlm,ylm[yn][:lmax+1,:lmax+1])/w0
            xb[i,yn,:] = basic.aps.cl2bcl(bn,lmax,xl[i,yn,:])
        np.savetxt(fx.xl[i],np.concatenate((bc[None,:],xb[i,:,:])).T)

    # save to file
    if rlz[-1] >= 2:
        print('save sim') 
        np.savetxt(fx.mxls,np.concatenate((l[None,:],np.mean(xl[1:,:,:],axis=0),np.std(xl[1:,:,:],axis=0))).T)
        np.savetxt(fx.mxbs,np.concatenate((bc[None,:],np.mean(xb[1:,:,:],axis=0),np.std(xb[1:,:,:],axis=0))).T)
        cov = np.cov(xb[1:,0,:],rowvar=0)
        np.savetxt(fx.xcov,cov)

    if rlz[0] == 0:
        print('save real')
        np.savetxt(fx.oxls,np.concatenate((l[None,:],xl[0,:,:],np.std(xl[1:,:,:],axis=0))).T)
        np.savetxt(fx.oxbs,np.concatenate((bc[None,:],xb[0,:,:],np.std(xb[1:,:,:],axis=0))).T)



def interface(run=['yalm','tauxy'],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},ep=1e-30):
    
    p = prjlib.init_analysis(**kwargs_cmb)
    W, M, wn = prjlib.set_mask(p.famask)

    # load obscls
    if p.fltr == 'none':
        Wt = W

    if p.fltr == 'cinv':
        Wt = M

    # define objects
    qtau, qlen, qtbh, qlbh = prjlib.init_quad(p.ids,p.stag,**kwargs_qrec)

    if 'yalm' in run:

        for ytype in ['nilc','milca']:

            for masktype in range(5):
    
                cy = prjlib.init_compy(p.ids,masktype=masktype,ytype=ytype)
                #Wy = hp.fitsfunc.read_map(cy.fymask,field=cy.masktype)
                Wy = hp.fitsfunc.read_map(cy.faymask,field=cy.masktype)
                w2 = np.average(Wy**2)
                ymap2yalm(cy,Wy,p.rlz,p.lmax,w2)

    if 'tauxy' in run:

        for ytype in ['nilc','milca']:
            
            for masktype in range(5):
        
                cy = prjlib.init_compy(p.ids,masktype=masktype,ytype=ytype)
                fxtau, fxtbh = prjlib.init_cross(qtau,qtbh,cy,p.ids,p.stag)

                # normalization
                #Wy = hp.fitsfunc.read_map(cy.fymask,field=cy.masktype)
                Wy = hp.fitsfunc.read_map(cy.faymask,field=cy.masktype)
                w0 = np.average(Wy*Wt**2)
                w1 = np.average(Wy**2)
                print('wind:',w0,w1)

                # xaps
                tauxy(cy,qtau.olmax,p.bn,p.rlz,qtau.f,fxtau,w0,w1)
                tauxy(cy,qtau.olmax,p.bn,p.rlz,qtbh.f,fxtbh,w0,w1)


