
import numpy as np
import healpy as hp
import scipy.signal as sp
import pickle
import os
from astropy.io import fits
import tqdm

#from cmblensplus/wrap/
import basic
import curvedsky

#from cmblensplus/utils/
import misctools

#local module
import prjlib


def save_beam(fsmap,fbeam,lmax):

    if not os.path.exists(fbeam):
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

        # convert to alm
        alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,hpmap)

        # beam deconvolution
        alm *= ibl[:,None]#/pfunc[:,None]

    else:
        # non-zero tau sim
        # deconvolve beam first
        hpmap = reduc_map(dtype,fmap,scale=scale)
        nside = hp.pixelfunc.get_nside(hpmap)
        alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,hpmap)
        alm *= ibl[:,None]
        # input signal amplitude modulation
        alm = curvedsky.utils.mulwin(alm,mask*np.exp(-tmap))
    
    # save to file
    pickle.dump((alm),open(falm,"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def map2alm_all(rlz,lmax,fmap,falm,wind,fbeam,dtype,sscale=1.,nscale=1.,stype=['s','n'],ftalm=None,**kwargs):

    # beam function
    ibl = 1./np.loadtxt(fbeam)[:lmax+1]

    for i in tqdm.tqdm(rlz,ncols=100,desc='map2alm:'):
        
        if i == 0:  # real data
            map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,**kwargs)
        
        else:  # simulation

            if ftalm is None:
                tmap = None
            else:
                nside = hp.pixelfunc.get_nside(wind)
                talm = pickle.load(open(ftalm[i],"rb"))
                tmap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,talm)
        
            if 's' in stype:  map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,scale=sscale,tmap=tmap,**kwargs)
            if 'n' in stype:  map2alm(lmax,fmap['n'][i],falm['n']['T'][i],wind,ibl,dtype,scale=nscale,**kwargs)


def gen_tau(rlz,lmax,ftalm,**kwargs_ov):
    
    for i in tqdm.tqdm(rlz,ncols=100,desc='generate tau alm:'):
        
        if misctools.check_path(ftalm[i],**kwargs_ov): continue
        
        if i==0: continue

        tt  = prjlib.tau_spec(lmax)
        alm = curvedsky.utils.gauss1alm(lmax,tt)
        pickle.dump((alm),open(ftalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm_comb(rlz,falm,stype=['n','p'],overwrite=False,verbose=True):

    for i in tqdm.tqdm(rlz,ncols=100,desc='alm combine:'):

        if misctools.check_path(falm['c']['T'][i],overwrite=overwrite,verbose=verbose): continue

        alm = pickle.load(open(falm['s']['T'][i],"rb"))
        nlm = 0.*alm
        plm = 0.*alm
        if i>0:
            if 'n' in stype:  nlm = pickle.load(open(falm['n']['T'][i],"rb"))
            if 'p' in stype:  plm = pickle.load(open(falm['p']['T'][i],"rb"))

        pickle.dump((alm+nlm+plm),open(falm['c']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def wiener_cinv_core(i,dtype,M,cl,bl,Nij,fmap,falm,sscale,nscale,ftalm,verbose=True,**kwargs):

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
            alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Ts) / bl[0,:,None]
            alm = curvedsky.utils.mulwin(alm,np.exp(-tmap)) # add tau effect
            Ts = M * curvedsky.utils.hp_alm2map(nside,lmax,lmax,alm*bl[0,:,None])
        
        # noise, ptsr
        Tn = M * reduc_map(dtype,fmap['n'][i],scale=nscale)
        #if 'nosz' in dtype:
        #    Tp = 0.
        #else:
        Tp = M * reduc_map(dtype,fmap['p'][i].replace('a0.0deg','a1.0deg'),TK=1.) # approximately use 1.0deg apodization case

        T[0,0,:] = Ts + Tn + Tp

    # cinv
    Tlm = curvedsky.cninv.cnfilter_freq(1,1,nside,lmax,cl,bl,Nij,T,filter='W',ro=10,verbose=verbose,**kwargs)

    pickle.dump((Tlm[0,:,:]),open(falm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def wiener_cinv(rlz,dtype,M,cl,fbeam,fmap,falm,sscale,nscale,ftalm=None,kwargs_ov={},kwargs_cinv={}):

    lmin = 1
    lmax = len(cl[0,:]) - 1

    bl  = np.reshape(np.loadtxt(fbeam)[:lmax+1],(1,lmax+1))
    Nij = M * (30.*(np.pi/10800.)/2.726e6)**(-2)
    Nij = np.reshape(Nij,(1,1,len(M)))

    for i in tqdm.tqdm(rlz,ncols=100,desc='wiener cinv:'):
        
        if ftalm is not None and i==0: continue  # avoid real tau case

        if misctools.check_path(falm[i],**kwargs_ov): continue

        wiener_cinv_core(i,dtype,M,cl,bl,Nij,fmap,falm,sscale,nscale,ftalm=ftalm,verbose=kwargs_ov['verbose'],**kwargs_cinv)



def alm2aps(rlz,lmax,fcmb,w2,stype=['s','n','p','c'],cli_out=True,**kwargs_ov):  # compute aps
    # output is ell, TT(s), TT(n), TT(p), TT(s+n+p)

    if misctools.check_path(fcmb.scl,**kwargs_ov):  return

    eL = np.linspace(0,lmax,lmax+1)
    cl = np.zeros((len(rlz),4,lmax+1))

    for ii, i in enumerate(tqdm.tqdm(rlz,ncols=100,desc='cmb alm2aps:')):

        if 's' in stype:  salm = pickle.load(open(fcmb.alms['s']['T'][i],"rb"))
        if i>0:
            if 'n' in stype:  nalm = pickle.load(open(fcmb.alms['n']['T'][i],"rb"))
            if 'p' in stype:  palm = pickle.load(open(fcmb.alms['p']['T'][i],"rb"))
            if 'c' in stype:  oalm = pickle.load(open(fcmb.alms['c']['T'][i],"rb"))

        #compute cls
        if 's' in stype:  cl[ii,0,:] = curvedsky.utils.alm2cl(lmax,salm) / w2
        if i>0:
            if 'n' in stype:  cl[ii,1,:] = curvedsky.utils.alm2cl(lmax,nalm) / w2
            if 'p' in stype:  cl[ii,2,:] = curvedsky.utils.alm2cl(lmax,palm) / w2
            if 'c' in stype:  cl[ii,3,:] = curvedsky.utils.alm2cl(lmax,oalm) / w2
                
        if cli_out:  np.savetxt(fcmb.cl[i],np.concatenate((eL[None,:],cl[ii,:,:])).T)

    # save to files
    if rlz[-1]>2:
        if kwargs_ov['verbose']:  print('cmb alm2aps: save sim')
        i0 = max(0,1-rlz[0])
        np.savetxt(fcmb.scl,np.concatenate((eL[None,:],np.mean(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)

    if rlz[0] == 0:
        if kwargs_ov['verbose']:  print('cmb alm2aps: save real')
        np.savetxt(fcmb.ocl,np.array((eL,cl[0,0,:])).T)



def gen_ptsr(rlz,fcmb,fbeam,fseed,fcl,fmap,w,olmax=2048,ilmin=1000,ilmax=3000,overwrite=False,verbose=True): # generating ptsr contributions

    # difference spectrum with smoothing
    scl = (np.loadtxt(fcmb.scl)).T[1][:ilmax+1]
    ncl = (np.loadtxt(fcmb.scl)).T[2][:ilmax+1]
    rcl = (np.loadtxt(fcmb.ocl)).T[1][:ilmax+1]

    # interpolate
    dCL = rcl - scl - ncl
    dcl = sp.savgol_filter(dCL, 101, 1)
    dcl[dcl<=0] = 1e-30
    dcl[:ilmin]  = 1e-30
    np.savetxt(fcl,np.array((np.linspace(0,ilmax,ilmax+1),dcl,dCL)).T)
    dcl = np.sqrt(dcl)

    # generating seed, only for the first run
    for i in rlz:
        if not os.path.exists(fseed[i]):
            alm = curvedsky.utils.gauss1alm(ilmax,np.ones(ilmax+1))
            pickle.dump((alm),open(fseed[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # load beam function
    bl = np.loadtxt(fbeam)[:ilmax+1]
    nside = hp.pixelfunc.get_nside(w)
    #pfunc = hp.sphtfunc.pixwin(nside)[:lmax+1]

    # multiply cl, transform to map and save it
    for i in tqdm.tqdm(rlz,ncols=100,desc='gen ptsr:'):

        if misctools.check_path(fcmb.alms['p']['T'][i],overwrite=overwrite,verbose=verbose): continue
        
        if i==0: continue
        
        palm = pickle.load(open(fseed[i],"rb"))
        palm *= dcl[:,None]*bl[:,None] #multiply beam-convolved cl
        pmap = curvedsky.utils.hp_alm2map(nside,ilmax,ilmax,palm)
        hp.fitsfunc.write_map(fmap['p'][i],pmap,overwrite=True)
        
        palm = curvedsky.utils.hp_map2alm(nside,olmax,olmax,w*pmap) #multiply window
        palm /= bl[:olmax+1,None]  #beam deconvolution
        pickle.dump((palm),open(fcmb.alms['p']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

        
def gen_nosz_noise(rlz,fcmb,fbeam,fnseed,fnl,fimap,lmax,TK=2.726,lmin=800,overwrite=False,verbose=True):

    rcl = (np.loadtxt(fcmb.ocl)).T[1]
    scl = (np.loadtxt(fcmb.scl)).T[1]
    lmax = len(rcl) - 1

    # interpolate
    dCL = rcl - scl
    dcl = sp.savgol_filter(dCL, 101, 1)
    dcl[dcl<=0] = 1e-30
    dcl[:lmin]  = 1e-30
    np.savetxt(fnl,np.array((np.linspace(0,lmax,lmax+1),dcl,dCL)).T)
    dcl = np.sqrt(dcl)
    
    # generating seed, only for the first run
    for i in rlz:
        if not os.path.exists(fnseed[i]):
            alm = curvedsky.utils.gauss1alm(lmax,np.ones(lmax+1))
            pickle.dump((alm),open(fnseed[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # load beam function
    bl = np.loadtxt(fbeam)[:lmax+1]
    nside = 2048
    
    # multiply cl, transform to map and save it
    for i in tqdm.tqdm(rlz,ncols=100,desc='gen noise for nosz:'):

        if misctools.check_path(fcmb.alms['n']['T'][i],overwrite=overwrite,verbose=verbose): continue

        if i==0: continue

        nalm = pickle.load(open(fnseed[i],"rb"))
        nalm *= dcl[:,None]*bl[:,None] #multiply beam-convolved cl
        nmap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,nalm) * TK
        hp.fitsfunc.write_map(fimap['n'][i],nmap,overwrite=True)
    

def interface(run=[],kwargs_cmb={},kwargs_ov={},kwargs_cinv={}):

    # define parameters, filenames and functions
    p = prjlib.init_analysis(**kwargs_cmb)

    # read survey window
    w, M, wn = prjlib.set_mask(p.famask)
    if p.fltr == 'cinv':  wn[:] = wn[0]

    # read beam function
    if p.dtype in ['dr2_nilc','dr2_smica','dr3_nilc','dr3_smica']:
        save_beam(p.fimap['s'][0],p.fbeam,p.lmax)

    # generate tau alm and save to ftalm file
    if 'tausim' in run and p.tausig:
        gen_tau(p.rlz,p.lmax,p.ftalm,**kwargs_ov)
    
    # generate ptsr
    if 'ptsr' in run and not p.tausig and p.dtype!='dr3_nosz':
        if p.dtype=='dr2_nilc':  ilmin, ilmax = 400, 3000
        if p.dtype=='dr2_smica': ilmin, ilmax = 1000, p.lmax
        # compute signal and noise spectra but need to change file names
        q = prjlib.init_analysis(**kwargs_cmb)
        q.fcmb.scl = q.fcmb.scl.replace('.dat','_tmp.dat')
        q.fcmb.ocl = q.fcmb.ocl.replace('.dat','_tmp.dat')
        if not misctools.check_path(q.fcmb.scl,**kwargs_ov):
            # compute signal and noise alms
            map2alm_all(p.rlz,ilmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,**kwargs_ov)
            alm2aps(p.rlz,ilmax,q.fcmb,wn[2],stype=['s','n'],cli_out=False,**kwargs_ov)
        # generate ptsr alm from obs - (sig+noi) spectrum
        gen_ptsr(p.rlz,q.fcmb,p.fbeam,p.fpseed,p.fptsrcl,p.fimap,w,olmax=p.lmax,ilmin=ilmin,ilmax=ilmax,**kwargs_ov) # generate map and alm from above computed aps

    # noise for nosz
    if 'ptsr' in run and p.dtype=='dr3_nosz': 
        # no-SZ simulation assumes signal + noise where signal is taken from smica DR2
        map2alm_all(p.rlz,p.lmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,stype=['s'],**kwargs_ov)
        q = prjlib.init_analysis(**kwargs_cmb)
        q.fcmb.scl = q.fcmb.scl.replace('.dat','_tmp.dat')
        q.fcmb.ocl = q.fcmb.ocl.replace('.dat','_tmp.dat')
        alm2aps(p.rlz,p.lmax,q.fcmb,wn[2],stype=['s'],**kwargs_ov)
        gen_nosz_noise(p.rlz,q.fcmb,p.fbeam,p.fnseed,p.fnosz_nl,p.fimap,p.lmax,**kwargs_ov)


    # use normal transform to alm
    if p.fltr == 'none':
        
        if p.dtype == 'dr3_nosz':
            stypes = ['s','n','c']
        else:
            stypes = ['s','n','p','c']
    
        if 'alm' in run:  # combine signal, noise and ptsr
            if p.tausig:
                # modify signal alm to include tau effect
                map2alm_all(p.rlz,p.lmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,ftalm=p.ftalm,**kwargs_ov)
            # convert map to alm
            #if p.dtype == 'dr3_nosz':
            map2alm_all(p.rlz,p.lmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,**kwargs_ov)
            # combine signal, noise and ptsr alms
            alm_comb(p.rlz,p.fcmb.alms,stype=stypes,**kwargs_ov)

        if 'aps' in run:  # compute cl
            alm2aps(p.rlz,p.lmax,p.fcmb,wn[2],stype=stypes,**kwargs_ov)

    # map -> alm with cinv filtering
    if p.fltr == 'cinv':  

        falm = p.fcmb.alms['c']['T']  #output file of cinv alms
    
        if 'alm' in run:  # cinv filtering here
            wiener_cinv(p.rlz,p.dtype,M,p.lcl[0:1,:p.lmax+1],p.fbeam,p.fimap,falm,p.sscale,p.nscale,ftalm=p.ftalm,kwargs_ov=kwargs_ov,kwargs_cinv=kwargs_cinv)

        if 'aps' in run:  # aps of filtered spectrum
            p.fcmb.alms['s']['T'] = falm # since cinv only save 'c', not 's'
            alm2aps(p.rlz,p.lmax,p.fcmb,wn[0],stype=['s','c'],**kwargs_ov)
    

