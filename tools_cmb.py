
import numpy as np
import healpy as hp
import scipy.signal as sp
import pickle
import os
from astropy.io import fits

#from cmblensplus/wrap/
import basic
import curvedsky

#from cmblensplus/utils/
import misctools

#local module
import prjlib


def get_beam(fsmap,fbeam,lmax):

    bl = fits.getdata(fsmap,extname='BEAMTF')
    np.savetxt(fbeam,bl)


def reduc_map(dtype,fmap,TK=2.726,field=0,scale=1.):

    if 'hm' in dtype:
        map1 = hp.fitsfunc.read_map(fmap[0],field=field,verbose=False)
        map2 = hp.fitsfunc.read_map(fmap[1],field=field,verbose=False)
        rmap = (map1-map2)*.5
    else:
        rmap = hp.fitsfunc.read_map(fmap,field=field,verbose=False)

    return rmap * scale / TK


def map2alm(lmax,fmap,falm,mask,ibl,dtype,scale=1.,tmap=None,**kwargs):

    if misctools.check_path(falm,**kwargs): return
    
    if tmap is None:
        # standard map -> alm conversion

        hpmap = reduc_map(dtype,fmap,scale=scale)
        nside = hp.pixelfunc.get_nside(hpmap)
        hpmap *= mask

        #pixel function
        #if pwind:
        #    pfunc = hp.sphtfunc.pixwin(nside)[:lmax+1]
        #else:
        #    pfunc = np.ones(lmax+1)
    
        # convert to alm
        alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,hpmap)

        # beam deconvolution
        alm *= ibl[:,None]#/pfunc[:,None]

    else:
        # deconvolve beam first
        hpmap = reduc_map(dtype,fmap,scale=scale)
        nside = hp.pixelfunc.get_nside(hpmap)
        alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,hpmap)
        alm *= ibl[:,None]
        # input signal amplitude modulation
        alm = curvedsky.utils.mulwin(nside,lmax,lmax,alm,mask*np.exp(-tmap))
    
    # save to file
    pickle.dump((alm),open(falm,"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def map2alm_all(rlz,lmax,fmap,falm,wind,fbeam,dtype,sscale=1.,nscale=1.,ftalm=None,**kwargs):

    # beam function
    ibl = 1./np.loadtxt(fbeam)[:lmax+1]

    for i in rlz:
        
        if kwargs.get('verbose'):  misctools.progress(i,rlz,text='Current progress',addtext='(map2alm_all)')
        
        if i == 0:
            map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,**kwargs)
        else:
            if ftalm is None:
                tmap = None
            else:
                nside = hp.pixelfunc.get_nside(wind)
                talm = pickle.load(open(ftalm[i],"rb"))
                tmap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,talm)
            map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,scale=sscale,tmap=tmap,**kwargs)
            map2alm(lmax,fmap['n'][i],falm['n']['T'][i],wind,ibl,dtype,scale=nscale,**kwargs)


def gen_tau(rlz,lmax,ftalm,**kwargs_ov):
    
    for i in rlz:
        
        if misctools.check_path(ftalm[i],**kwargs_ov): continue
        
        if i==0: continue

        tt  = prjlib.tau_spec(lmax)
        alm = curvedsky.utils.gauss1alm(lmax,tt)
        pickle.dump((alm),open(ftalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm_comb(rlz,falm,overwrite=False,verbose=True):

    for i in rlz:

        if misctools.check_path(falm['c']['T'][i],overwrite=overwrite,verbose=verbose): continue
        if verbose:  misctools.progress(i,rlz,text='Current progress',addtext='(alm_comb)')

        alm  = pickle.load(open(falm['s']['T'][i],"rb"))

        if i>0:
            nlm = pickle.load(open(falm['n']['T'][i],"rb"))
            plm = pickle.load(open(falm['p']['T'][i],"rb"))
        else:
            nlm = 0.*alm
            plm = 0.*alm

        pickle.dump((alm+nlm+plm),open(falm['c']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def wiener_cinv_core(i,dtype,M,cl,bl,Nij,fmap,falm,sscale,nscale,ftalm,**kwargs):

    lmax  = len(cl[0,:]) - 1
    
    nside = hp.pixelfunc.get_nside(M)
    npix  = 12*nside**2

    T  = np.zeros((1,1,npix))

    if i==0: 
        T[0,0,:] = M * reduc_map(dtype,fmap['s'][i])
    else:
        # signal
        Ts = reduc_map(dtype,fmap['s'][i],scale=sscale)
        if ftalm is None:
            Ts *= M
        else:
            talm = pickle.load(open(ftalm[i],"rb"))
            tmap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,talm)
            alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Ts) / bl[:,None]
            alm = curvedsky.utils.mulwin(nside,lmax,lmax,alm,np.exp(-tmap)) # add tau effect
            Ts = M * curvedsky.utils.hp_alm2map(nside,lmax,lmax,alm*bl[:,None])
        
        # noise, ptsr
        Tn = M * reduc_map(dtype,fmap['n'][i],scale=nscale)
        Tp = M * reduc_map(dtype,fmap['p'][i].replace('a0.0deg','a1.0deg'),TK=1.) # approximately use 1.0deg apodization case

        T[0,0,:] = Ts + Tn + Tp

    # cinv
    Tlm = curvedsky.cninv.cnfilter_freq(1,1,nside,lmax,cl,bl,Nij,T,filter='W',ro=10,**kwargs)

    pickle.dump((Tlm[0,:,:]),open(falm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def wiener_cinv(rlz,dtype,M,cl,fbeam,fnij,fmap,falm,sscale,nscale,ftalm=None,kwargs_ov={},kwargs_cinv={}):

    lmin = 1
    lmax = len(cl[0,:]) - 1

    bl  = np.reshape(np.loadtxt(fbeam)[:lmax+1],(1,lmax+1))
    Nij = M * (30.*(np.pi/10800.)/2.726e6)**(-2)
    Nij = np.reshape(Nij,(1,1,len(M)))

    for i in rlz:
        
        if ftalm is not None and i==0: continue  # avoid real tau case

        if misctools.check_path(falm[i],**kwargs_ov): continue
        if kwargs_ov.get('verbose'):  misctools.progress(i,rlz,text='Current progress',addtext='(wiener_cinv)')

        wiener_cinv_core(i,dtype,M,cl,bl,Nij,fmap,falm,sscale,nscale,ftalm=ftalm,**kwargs_cinv)



def alm2aps(rlz,lmax,fcmb,w2,stype=['s','n','p','c'],overwrite=False,verbose=True):  # compute aps
    # output is ell, TT(s), TT(n), TT(p), TT(s+n+p)

    if misctools.check_path(fcmb.scl,overwrite=overwrite,verbose=verbose):  return

    eL  = np.linspace(0,lmax,lmax+1)
    cls = np.zeros((len(rlz),4,lmax+1))

    for i in rlz:

        if verbose:  misctools.progress(i,rlz,text='Current progress',addtext='(aps cmb)')

        if 's' in stype:  salm = pickle.load(open(fcmb.alms['s']['T'][i],"rb"))
        if i>0:
            if 'n' in stype:  nalm = pickle.load(open(fcmb.alms['n']['T'][i],"rb"))
            if 'p' in stype:  palm = pickle.load(open(fcmb.alms['p']['T'][i],"rb"))
            if 'c' in stype:  oalm = pickle.load(open(fcmb.alms['c']['T'][i],"rb"))

        #compute cls
        ii = i - min(rlz)
        if 's' in stype:  cls[ii,0,:] = curvedsky.utils.alm2cl(lmax,salm) / w2
        if i>0:
            if 'n' in stype:  cls[ii,1,:] = curvedsky.utils.alm2cl(lmax,nalm) / w2
            if 'p' in stype:  cls[ii,2,:] = curvedsky.utils.alm2cl(lmax,palm) / w2
            if 'c' in stype:  cls[ii,3,:] = curvedsky.utils.alm2cl(lmax,oalm) / w2

    # save to files
    if rlz[-1] >= 2:
        if verbose:  print('save sim')
        i0 = max(0,1-rlz[0])
        np.savetxt(fcmb.scl,np.concatenate((eL[None,:],np.mean(cls[i0:,:,:],axis=0),np.std(cls[i0:,:,:],axis=0))).T)

    if rlz[0] == 0:
        if verbose:  print('save real')
        np.savetxt(fcmb.ocl,np.array((eL,cls[0,0,:])).T)



def gen_ptsr(rlz,fcmb,fbeam,fseed,fcl,fmap,w,lmin=1000,overwrite=False,verbose=True): # generating ptsr contributions

    #if p.dtype=='nilc': lmin = 800

    # difference spectrum with smoothing
    scl = (np.loadtxt(fcmb.scl)).T[1]
    ncl = (np.loadtxt(fcmb.scl)).T[2]
    rcl = (np.loadtxt(fcmb.ocl)).T[1]
    lmax = len(rcl) - 1

    # interpolate
    dCL = rcl - scl - ncl
    dcl = sp.savgol_filter(dCL, 101, 1)
    dcl[dcl<=0] = 1e-30
    dcl[:lmin]  = 1e-30
    np.savetxt(fcl,np.array((np.linspace(0,lmax,lmax+1),dcl,dCL)).T)
    dcl = np.sqrt(dcl)

    # generating seed, only for the first run
    for i in rlz:
        if not os.path.exists(fseed[i]):
            alm = curvedsky.utils.gauss1alm(lmax,np.ones(lmax+1))
            pickle.dump((alm),open(fseed[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # load beam function
    bl = np.loadtxt(fbeam)[:lmax+1]
    nside = hp.pixelfunc.get_nside(w)
    #pfunc = hp.sphtfunc.pixwin(nside)[:lmax+1]

    # multiply cl, transform to map and save it
    for i in rlz:

        if misctools.check_path(fcmb.alms['p']['T'][i],overwrite=overwrite,verbose=verbose): continue
        
        if i==0: continue
        if verbose:  misctools.progress(i,rlz,text='Current progress',addtext='(gen_ptsr)')
        
        palm = pickle.load(open(fseed[i],"rb"))
        palm *= dcl[:,None]*bl[:,None] #multiply beam-convolved cl
        pmap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,palm)
        hp.fitsfunc.write_map(fmap['p'][i],pmap,overwrite=True)
        
        palm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,w*pmap) #multiply window
        palm /= bl[:,None]  #beam deconvolution
        pickle.dump((palm),open(fcmb.alms['p']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def interface(run=[],kwargs_cmb={},kwargs_ov={},kwargs_cinv={}):

    # define parameters, filenames and functions
    p = prjlib.init_analysis(**kwargs_cmb)

    # read survey window
    w, M, wn = prjlib.set_mask(p.famask)
    if p.fltr == 'cinv':  wn[:] = wn[0]

    # read beam function
    if p.dtype in ['dr2_nilc','dr2_smica','dr3_nilc','dr3_smica']:
        get_beam(p.fimap['s'][0],p.fbeam,p.lmax)

    if 'tausim' in run and p.tausig:
        # generate tau alm and save to ftalm file
        gen_tau(p.rlz,p.lmax,p.ftalm,**kwargs_ov)
    
    if 'ptsr' in run and not p.tausig:  # generate ptsr
        # compute signal and noise alms
        map2alm_all(p.rlz,p.lmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,**kwargs_ov)
        # compute signal and noise spectra but need to change file names
        q = prjlib.init_analysis(**kwargs_cmb)
        q.fcmb.scl = q.fcmb.scl.replace('.dat','_tmp.dat')
        q.fcmb.ocl = q.fcmb.ocl.replace('.dat','_tmp.dat')
        alm2aps(p.rlz,p.lmax,q.fcmb,wn[2],stype=['s','n'],**kwargs_ov)
        # generate ptsr alm from obs - (sig+noi) spectrum
        gen_ptsr(p.rlz,q.fcmb,p.fbeam,p.fpseed,p.fptsrcl,p.fimap,w,**kwargs_ov) # generate map and alm from above computed aps

    if p.fltr == 'none':  # use normal transform to alm
    
        if 'alm' in run:  # combine signal, noise and ptsr
            if p.tausig:
                # modify signal alm to include tau effect
                map2alm_all(p.rlz,p.lmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,ftalm=p.ftalm,**kwargs_ov)
            # combine signal, noise and ptsr alms 
            alm_comb(p.rlz,p.fcmb.alms,**kwargs_ov)

        if 'aps' in run:  # compute cl
            alm2aps(p.rlz,p.lmax,p.fcmb,wn[2],overwrite=True,verbose=kwargs_ov['verbose'])

    if p.fltr == 'cinv':  # cinv filtering
    
        if 'alm' in run:  # cinv filtering here
            wiener_cinv(p.rlz,p.dtype,M,p.lcl[0:1,:p.lmax+1],p.fbeam,p.fcmb.fnij,p.fimap,p.fcmb.alms['c']['T'],p.sscale,p.nscale,ftalm=p.ftalm,kwargs_ov=kwargs_ov,kwargs_cinv=kwargs_cinv)

        if 'aps' in run:  # aps of filtered spectrum
            p.fcmb.alms['s']['T'] = p.fcmb.alms['c']['T']
            alm2aps(p.rlz,p.lmax,p.fcmb,wn[0],stype=['s','c'],**kwargs_ov)
    

