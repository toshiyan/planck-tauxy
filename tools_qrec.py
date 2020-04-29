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

# local
import prjlib


def aps(rlz,qobj,fklm=None,q='TT',overwrite=False,verbose=True):

    if misctools.check_path(qobj.f[q].ocls,overwrite=overwrite,verbose=verbose):  return

    cl = np.zeros((len(rlz),3,qobj.olmax+1))

    for i in rlz:
        
        if misctools.check_path(qobj.f[q].cl[i],**kwargs_ov): continue
        if verbose:  misctools.progress(i,rlz,text='Current progress',addtext='(qrec aps)')
        
        # load qlm
        alm = pickle.load(open(qobj.f[q].alm[i],"rb"))[0]
        mf  = pickle.load(open(qobj.f[q].mfb[i],"rb"))[0]
        alm -= mf

        # flip sign as e^-tau ~ 1-tau
        if qobj.qtype == 'tau':
            alm = -alm
        
        # auto spectrum
        ii = i - min(rlz)
        cl[ii,0,:] = curvedsky.utils.alm2cl(qobj.olmax,alm)/qobj.wn[4]

        if fklm is not None and i>0:
            # load input klm
            if qobj.qtype == 'lens':
                iKlm = hp.fitsfunc.read_alm(fklm[i])
                iklm = curvedsky.utils.lm_healpy2healpix(len(iKlm),iKlm,2048)        
            if qobj.qtype == 'tau':
                iklm = pickle.load(open(fklm[i],"rb"))
            # cross with input
            cl[ii,1,:] = curvedsky.utils.alm2cl(qobj.olmax,alm,iklm)/qobj.wn[2]
            # input
            cl[ii,2,:] = curvedsky.utils.alm2cl(qobj.olmax,iklm)

        np.savetxt(qobj.f[q].cl[i],np.concatenate((qobj.l[None,:],cl[ii,:,:])).T)

    # save to file
    if rlz[-1] >= 2:
        print('save sim') 
        i0 = max(0,1-min(rlz))
        np.savetxt(qobj.f[q].mcls,np.concatenate((qobj.l[None,:],np.mean(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)



def qrec_bh_tau(qtau,qlen,qtbh,rlz,q='TT',**kwargs_ov):

    At = np.loadtxt(qtau.f[q].al,unpack=True)[1]
    Ag = np.loadtxt(qlen.f[q].al,unpack=True)[1]

    # calculate response function
    if not misctools.check_path(qtbh.f[q].al,**kwargs_ov):

        Rtl = curvedsky.norm_lens.ttt(qtau.olmax,qtau.rlmin,qtau.rlmax,qtau.lcl[0,:qtau.rlmax+1],qtau.ocl[0,:qtau.rlmax+1],gtype='k')
        Il = 1./(1.-At*Ag*Rtl**2)
        np.savetxt(qtbh.f[q].al,np.array((qtau.l,At*Il,At)).T)
    

    # calculate bh-estimator 
    for i in rlz:
        
        if misctools.check_path(qtbh.f[q].alm[i],**kwargs_ov): continue
        if kwargs_ov['verbose']:  misctools.progress(i,rlz,text='Current progress',addtext='(qrec_bh_tau)')

        tlm = pickle.load(open(qtau.f[q].alm[i],"rb"))[0]
        glm = pickle.load(open(qlen.f[q].alm[i],"rb"))[0]
        alm = ( tlm - At[:,None]*Rtl[:,None]*glm ) * Il[:,None]
        
        pickle.dump((alm,alm*0.),open(qtbh.f[q].alm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    quad_func.quad.mean_rlz(qtbh,rlz,**kwargs_ov)



def interface(qrun=['norm','qrec','n0','mean'],run=['tau','len','tbh'],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},ep=1e-30):
    
    p = prjlib.init_analysis(**kwargs_cmb)
    __, __, wn = prjlib.set_mask(p.famask)

    # load obscls
    if p.fltr == 'none':
        ocl = np.loadtxt(p.fcmb.scl,unpack=True)[4:5]
        ifl = ocl.copy()

    if p.fltr == 'cinv':
        bl  = np.loadtxt(p.fbeam)[:p.lmax+1]
        cnl = p.lcl[0,:] + (1./bl)**2*(30.*np.pi/10800./2.72e6)**2
        wcl = np.loadtxt(p.fcmb.scl,unpack=True)[4]

        # quality factor defined in Planck 2015 lensing paper
        Ql  = (p.lcl[0,:])**2/(wcl*cnl+ep**2)
        # T' = QT^f = Q/(cl+nl) * (T+n)/sqrt(Q)

        ocl = cnl/(Ql+ep)  # corrected observed cl
        ifl = p.lcl[0,:]/(Ql+ep)    # remove theory signal cl in wiener filter
        ocl = np.reshape(ocl,(1,p.lmax+1))
        ifl = np.reshape(ifl,(1,p.lmax+1))

        wn[:] = wn[0]

    # define objects
    qtau, qlen, qtbh, qlbh = prjlib.init_quad(p.ids,p.stag,wn=wn,lcl=p.lcl,ocl=ocl,ifl=ifl,falm=p.fcmb.alms['c'],**kwargs_qrec)

    # reconstruction
    if 'tau' in run:
        quad_func.qrec_flow(qtau,p.rlz,run=qrun,**kwargs_ov)  #tau rec
        if 'aps' in qrun:  aps(p.rlz,qtau,fklm=p.ftalm,**kwargs_ov)

    if 'len' in run:
        quad_func.qrec_flow(qlen,p.rlz,run=qrun,**kwargs_ov)  #lens rec
        if 'aps' in qrun:  aps(p.rlz,qlen,fklm=p.fiklm,**kwargs_ov)

    if 'tbh' in run:
        qrec_bh_tau(qtau,qlen,qtbh,p.rlz,**kwargs_ov)  #BHE for tau
        if 'aps' in qrun:  aps(p.rlz,qtbh,fklm=p.ftalm,**kwargs_ov)


