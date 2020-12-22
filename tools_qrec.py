# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle
import tqdm

# from cmblensplus/wrap
import curvedsky
import basic
import binning as bn

# from cmblensplus/utils
import quad_func
import misctools

# local
import prjlib


def init_quad(ids,stag,rlz=[],**kwargs):
    
    d = prjlib.data_directory()
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    qtau = quad_func.quad(rlz=rlz,stag=stag,root=d['root'],ids=ids,qtype='tau',**kwargs)
    qlen = quad_func.quad(rlz=rlz,stag=stag,root=d['root'],ids=ids,qtype='lens',**kwargs)
    qsrc = quad_func.quad(rlz=rlz,stag=stag,root=d['root'],ids=ids,qtype='src',**kwargs)
    qtbh = quad_func.quad(rlz=rlz,stag=stag,root=d['root'],ids=ids,qtype='tau',bhe=['lens'],**kwargs)
    qtBH = quad_func.quad(rlz=rlz,stag=stag,root=d['root'],ids=ids,qtype='tau',bhe=['lens','src'],**kwargs)


    #qtau.fname(d['root'],ids,stag)
    #qlen.fname(d['root'],ids,stag)
    #qsrc.fname(d['root'],ids,stag)
    #qtbh.fname(d['root'],ids,stag)
    #qtBH.fname(d['root'],ids,stag)

    return qtau, qlen, qsrc, qtbh, qtBH



def aps(rlz,qobj,fklm=None,q='TT',**kwargs_ov):

    cl = np.zeros((len(rlz),3,qobj.olmax+1))
    
    for i in tqdm.tqdm(rlz,ncols=100,desc='aps ('+qobj.qtype+')'):
        
        if misctools.check_path(qobj.f[q].cl[i],**kwargs_ov): continue
        
        # load qlm
        alm = pickle.load(open(qobj.f[q].alm[i],"rb"))[0]
        mf  = pickle.load(open(qobj.f[q].mfb[i],"rb"))[0]
        alm -= mf

        # flip sign as e^-tau ~ 1-tau
        if qobj.qtype == 'tau':
            alm = -alm
        
        # auto spectrum
        cl[i,0,:] = curvedsky.utils.alm2cl(qobj.olmax,alm)/qobj.wn[4]

        if fklm is not None and i>0:
            # load input klm
            if qobj.qtype == 'lens':
                iKlm = hp.fitsfunc.read_alm(fklm[i])
                iklm = curvedsky.utils.lm_healpy2healpix(iKlm,2048)        
            if qobj.qtype == 'tau':
                iklm = pickle.load(open(fklm[i],"rb"))
            if qobj.qtype == 'src':
                iklm = 0.*alm
            # cross with input
            cl[i,1,:] = curvedsky.utils.alm2cl(qobj.olmax,alm,iklm)/qobj.wn[2]
            # input
            cl[i,2,:] = curvedsky.utils.alm2cl(qobj.olmax,iklm)

        np.savetxt(qobj.f[q].cl[i],np.concatenate((qobj.l[None,:],cl[i,:,:])).T)

    # save to files
    if rlz[-1] >= 2:
        i0 = max(0,1-rlz[0])
        np.savetxt(qobj.f[q].mcls,np.concatenate((qobj.l[None,:],np.average(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)


def qrec_bh_tau(qtau,qlen,qsrc,qtbh,rlz,q='TT',est=['lens','tau','src'],**kwargs_ov):

    At = np.loadtxt(qtau.f[q].al,unpack=True)[1]

    # calculate response function
    Btt, Btg, Bts = quad_func.quad.coeff_bhe(qtau,est=est,qcomb=q,gtype='k')

    # normalization for BH
    if not misctools.check_path(qtbh.f[q].al,**kwargs_ov):
        np.savetxt(qtbh.f[q].al,np.array((qtau.l,At*Btt,At)).T)

    # calculate bh-estimator 
    for i in tqdm.tqdm(rlz,ncols=100,desc='forming bh-est'):
        
        if misctools.check_path(qtbh.f[q].alm[i],**kwargs_ov): continue

        tlm = pickle.load(open(qtau.f[q].alm[i],"rb"))[0]
        glm = pickle.load(open(qlen.f[q].alm[i],"rb"))[0]
        if 'src' in est: 
            slm = pickle.load(open(qsrc.f[q].alm[i],"rb"))[0]
        else:
            slm = 0.*glm

        alm = Btt[:,None]*tlm + Btg[:,None]*glm + Bts[:,None]*slm
        pickle.dump((alm,alm*0.),open(qtbh.f[q].alm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # calculate mean-field bias
    quad_func.quad.mean_rlz(qtbh,rlz,**kwargs_ov)


def load_binned_tt(mb,dtype='dr2_smica',fltr='cinv',cmask='Lmask',bhe=['lens'],snmax=100):
    
    # filename
    d = prjlib.data_directory()
    p = prjlib.init_analysis(snmax=snmax,dtype=dtype,fltr=fltr,wtype=cmask)
    qobj = quad_func.quad(stag=p.stag,root=d['root'],ids=p.ids,qtype='tau',bhe=bhe,rlmin=100,rlmax=2048)

    # optimal filter
    al = (np.loadtxt(qobj.f['TT'].al)).T[1]
    vl = al/np.sqrt(qobj.l+1e-30)
    
    # binned spectra
    mtt, __, stt, ott = bn.binned_spec(mb,qobj.f['TT'].cl,cn=1,doreal=True,opt=True,vl=vl)
    
    # noise bias
    nb = bn.binning( (np.loadtxt(qobj.f['TT'].n0bs)).T[1], mb, vl=vl )
    rd = np.array( [ (np.loadtxt(qobj.f['TT'].rdn0[i])).T[1] for i in p.rlz ] )
    rb = bn.binning(rd,mb,vl=vl)
    
    # debias
    ott = ott - rb[0] - nb/(qobj.mfsim)
    stt = stt - rb[1:,:] - nb/(qobj.mfsim-1)
    
    # sim mean and std
    mtt = mtt - np.mean(rb[1:,:],axis=0) - nb/(qobj.mfsim-1)
    vtt = np.std(stt,axis=0)
    
    # subtract average of sim
    ott = ott - mtt 
    
    return mtt, vtt, stt, ott


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

        ocl = cnl/(Ql+ep)  # corrected observed cl  = Fl
        ifl = p.lcl[0,:]/(Ql+ep)    # remove theory signal cl in wiener filter
        ocl = np.reshape(ocl,(1,p.lmax+1))
        ifl = np.reshape(ifl,(1,p.lmax+1))

        wn[:] = wn[0]

    # define objects
    qtau, qlen, qsrc, qtbh, qtBH = init_quad(p.ids,p.stag,rlz=p.rlz,wn=wn,lcl=p.lcl,ocl=ocl,ifl=ifl,falm=p.fcmb.alms['c'],**kwargs_qrec,**kwargs_ov)

    # reconstruction
    if 'tau' in run:
        qtau.qrec_flow(run=qrun)  #tau rec
        if 'aps' in qrun:  aps(p.rlz,qtau,fklm=p.ftalm,**kwargs_ov)

    if 'len' in run:
        qlen.qrec_flow(run=qrun)  #lens rec
        if 'aps' in qrun:  aps(p.rlz,qlen,fklm=p.fiklm,**kwargs_ov)

    if 'src' in run:
        qsrc.qrec_flow(run=qrun)  #src rec
        if 'aps' in qrun:  aps(p.rlz,qsrc,fklm=p.ftalm,**kwargs_ov)

    if 'tbh' in run:
        qtbh.qrec_flow(run=qrun)
        #qrec_bh_tau(qtau,qlen,qsrc,qtbh,p.rlz,est=['lens','tau'],**kwargs_ov)  #BHE for tau
        if 'aps' in qrun:  aps(p.rlz,qtbh,fklm=p.ftalm,**kwargs_ov)

    if 'tBH' in run:
        qtBH.qrec_flow(run=qrun)
        #qrec_bh_tau(qtau,qlen,qsrc,qtBH,p.rlz,est=['lens','tau','src'],**kwargs_ov)  #full BHE for tau
        if 'aps' in qrun:  aps(p.rlz,qtBH,fklm=p.ftalm,**kwargs_ov)

