# Local modules

# from external
import numpy as np
import healpy as hp
import sys
import configparser
import pickle

# from cmblensplus/wrap/
import curvedsky
import basic

# from cmblensplus/utils/
import constants
import quad_func
import misctools

# from pylib
import planck_filename as plf


def data_directory():
    
    direct = {}

    root = '/global/cscratch1/sd/toshiyan/plk/'
    #root = '/global/u1/t/toshiyan/scratch/plk/'
    #root = '/global/u1/t/toshiyan/scratch/plk_old/'
    #direct['dr2']  = '/project/projectdirs/cmb/data/planck2015/'
    #direct['dr3']  = '/project/projectdirs/cmb/data/planck2018/'
    #direct['PR2L'] = '/global/cscratch1/sd/toshiyan/PR2/'
    #direct['PR3L'] = '/global/cscratch1/sd/toshiyan/PR3/'
    direct['root'] = root
    direct['inp']  = root + 'input/'
    direct['win']  = root + 'mask/'
    direct['cmb']  = root + 'cmb/'
    direct['bem']  = root + 'beam/'
    direct['ysz']  = root + 'ysz/'
    direct['xcor'] = root + 'xcor/'
    direct['ptsr'] = root + 'ptsr/'
    direct['nosz'] = root + 'nosz/'

    return direct



# * Define CMB file names
class cmb:

    def __init__(self,p,mtype=['T']):

        # set directory
        d = data_directory()
        d_alm = d['cmb'] + 'alm/'
        d_aps = d['cmb'] + 'aps/'
        d_map = d['cmb'] + 'map/'

        # cmb signal/noise alms
        self.alms = {}
        for s in ['s','n','p','c']:
            self.alms[s] = {}
            if s in ['n','p']: tag = p.ntag
            if s in ['s','c']: tag = p.stag
            for m in mtype:
                self.alms[s][m] = [d_alm+'/'+s+'_'+m+'_'+tag+'_'+x+'.pkl' for x in p.ids]

        # noise covariance
        self.fnij = d_map + 'nij_'+p.ntag+'.pkl'

        # cmb aps
        self.scl = d_aps+'aps_sim_1d_'+p.stag+'.dat'
        self.cl  = [d_aps+'/rlz/cl_'+p.stag+'_'+x+'.dat' for x in p.ids]
        self.xcl = d_aps+'aps_sim_1d_'+p.stag+'_cross.dat'
        self.ocl = d_aps+'aps_obs_1d_'+p.stag+'.dat'


#* Define parameters
class analysis:

    def __init__(self,snmin=0,snmax=100,dtype='dr2_smica',fltr='none',lmin=1,lmax=2048,olmin=1,olmax=2048,bn=30,wtype='Lmask',ascale=1.,tausig=False):

        #//// load config file ////#
        conf = misctools.load_config('CMB')

        # rlz
        self.snmin = conf.getint('snmin',snmin)
        self.snmax = conf.getint('snmax',snmax)
        self.snum  = self.snmax - self.snmin + 1
        self.rlz   = np.linspace(self.snmin,self.snmax,self.snum,dtype=np.int)

        # multipole of converted CMB alms
        self.lmin   = conf.getint('lmin',lmin)
        self.lmax   = conf.getint('lmax',lmax)

        # multipole of output CMB spectrum
        self.olmin  = conf.getint('olmin',olmin)
        self.olmax  = conf.getint('olmax',olmax)
        self.bn     = conf.getint('bn',bn)
        self.binspc = conf.get('binspc','')

        # cmb map
        self.dtype  = conf.get('dtype',dtype)
        self.tausig = conf.getboolean('tausig',tausig)
        self.fltr   = conf.get('fltr',fltr)
        if self.fltr == 'cinv':
            ascale = 0.

        # window
        self.wtype  = conf.get('wtype',wtype)
        if self.wtype == 'Fullsky':  
            ascale = 0.
            self.fltr == 'nonse'
        self.ascale = conf.getfloat('ascale',ascale)

        # cmb scaling (correction to different cosmology)
        self.sscale = 1.
        if self.dtype in ['dr2_nilc','dr2_smicaffp8']:  
            self.sscale = 1.0134 # suggested by Planck team

        # noise scaling (correction to underestimate of noise in sim)
        self.nscale = 1.
        if self.dtype!='dr2_nilc' and self.dtype!='dr3_nosz':  
            self.nscale = 1.03 # derived by comparing between obs and sim spectra of HM1-HM2


    def filename(self):

        #//// root directories ////#
        d = data_directory()

        #//// basic tags ////#
        # for alm
        apotag = 'a'+str(self.ascale)+'deg'
        if self.tausig: 
            self.stag = '_'.join( [ self.dtype , self.wtype , apotag , self.fltr , 'tausig' ] )
        else: 
            self.stag = '_'.join( [ self.dtype , self.wtype , apotag , self.fltr ] )
        self.ntag = '_'.join( [ self.dtype , self.wtype , apotag , self.fltr ] )

        # output multipole range
        self.otag = '_oL'+str(self.olmin)+'-'+str(self.olmax)+'_b'+str(self.bn)

        #//// index ////#
        self.ids = [str(i).zfill(5) for i in range(-1,100)]
        self.ids[0] = 'real'  # change 1st index
        ids = self.ids

        #//// Input public maps ////#
        # sim
        PLK = d[self.dtype[:3]]
        self.fimap = {}

        # PLANCK DR2
        if 'dr2' in self.dtype:
            self.fimap = plf.load_iqu_filename(PR=2,freq=self.dtype.replace('dr2_',''))
        if 'dr3' in self.dtype:
            self.fimap = plf.load_iqu_filename(PR=3,freq=self.dtype.replace('dr3_',''))

        '''
        if self.dtype in ['dr2_nilc','dr2_smicaffp8']:
            self.fimap['s']  = [PLK+'ffp8/compsep/mc_cmb/ffp8_'+self.dtype.replace('dr2_','')+'_int_cmb_mc_'+x+'_005a_2048.fits' for x in ids]
            self.fimap['n']  = [PLK+'ffp8/compsep/mc_noise/ffp8_'+self.dtype.replace('dr2_','')+'_int_noise_mc_'+x+'_005a_2048.fits' for x in ids]
        if self.dtype == 'dr2_smica':
            self.fimap['s']  = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_cmb_new_mc_'+x+'_005a_2048.fits' for x in ids]
            self.fimap['n']  = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_noise_mc_'+x+'_005a_2048.fits' for x in ids]        
        if self.dtype == 'dr3_nosz':
            #PLK = d['dr2']
            #self.fimap['s']  = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_cmb_new_mc_'+x+'_005a_2048.fits' for x in ids]
            #self.fimap['n']  = [d['nosz']+'noise_'+x+'.fits' for x in ids]
            self.fimap['s']  = [d['PR3L']+'cmb/sim/dx12_v3_smica_nosz_cmb_mc_'+x+'_raw.fits' for x in ids]
            self.fimap['n']  = [d['PR3L']+'cmb/sim/dx12_v3_smica_nosz_noise_mc_'+x+'_raw.fits' for x in ids] 
        if self.dtype == 'dr2_smicahm': # smica half mission
            smap1 = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_cmb_new_hm1_mc_'+x+'_005a_2048.fits' for x in ids]
            smap2 = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_cmb_new_hm2_mc_'+x+'_005a_2048.fits' for x in ids]
            nmap1 = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_noise_hm1_mc_'+x+'_005a_2048.fits' for x in ids]
            nmap2 = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_noise_hm2_mc_'+x+'_005a_2048.fits' for x in ids]
            self.fimap['s'] = [smap1,smap2]
            self.fimap['n'] = [nmap1,nmap2]
        # replace 1st rlz to real data
        if self.dtype in ['dr2_nilc','dr2_smica']:
            self.fimap['s'][0] = PLK+'pr2/cmbmaps/COM_CMB_IQU-'+self.dtype.replace('dr2_','')+'-field-Int_2048_R2.01_full.fits'        
        if self.dtype == 'dr2_smicahm':
            self.fimap['s'][0][0] = PLK+'pr2/cmbmaps/COM_CMB_IQU-smica-field-Int_2048_R2.01_halfmission-1.fits'
            self.fimap['s'][0][1] = PLK+'pr2/cmbmaps/COM_CMB_IQU-smica-field-Int_2048_R2.01_halfmission-2.fits'
        if self.dtype == 'dr2_smicaffp8': #same as dr2_smica
            self.fimap['s'][0] = PLK+'pr2/cmbmaps/COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits'        
        if self.dtype == 'dr3_nosz':
            self.fimap['s'][0] = d['inp']+'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits'
        '''

        # input klm realizations
        self.fiklm = [ plf.subd['pr2']['lens'] + 'sky_klms/sim_'+x[1:]+'_klm.fits' for x in ids ]

        # input tlm realizations
        if self.tausig:
            self.ftalm = [ d['inp'] + 'sky_tlms/sim_tausig_'+x[1:]+'_tlm.fits' for x in ids ]
        else:
            self.ftalm = None

        #//// base best-fit cls ////#
        # aps of Planck 2015 best fit cosmology
        self.flcl = plf.subd['pr2']['cosmo'] + 'COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt'
        
        # for forecast
        self.simcl = d['inp'] + 'forecast_cosmo2017_10K_acc3_lensedCls.dat'
        self.simul = d['inp'] + 'forecast_cosmo2017_10K_acc3_scalCls.dat'
        self.thocl = d['inp'] + 'forecast_tt_TH_R10.0_a0.0.dat'

        #//// Derived data filenames ////#
        # cmb map, alm and aps
        self.fcmb = cmb(self)

        # beam
        self.fbeam = d['bem'] + self.dtype+'.dat'
        if self.dtype == 'dr3_nosz': 
            self.fbeam = d['bem'] + 'dr2_smica.dat'

        # extra gaussian random fields
        #if self.dtype == 'dr3_nosz':
            # for nosz noise
        #    self.fnseed  = [d['nosz']+'seed_'+x+'.fits' for x in ids]
        #    self.fnosz_nl = d['nosz']+'noise_aps.dat'
        #    self.fimap['p'] = [x for x in ids]
        #else:
        # ptsr seed
        self.fpseed  = [d['ptsr']+x+'.fits' for x in ids]
        ptag = self.wtype+'_a'+str(self.ascale)+'deg'
        self.fptsrcl = d['ptsr']+'ptsr_'+ptag+'.dat'
        self.fimap['p'] = [d['ptsr']+'ptsr_'+ptag+'_'+x+'.fits' for x in ids]


    def set_mask_filename(self):
        
        d = data_directory()
        dwin = d['win']
        
        # this file contains several Gfsky
        self.fmask_Gfsky = dwin + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits' 
        
        # set mask filename
        if   self.wtype == 'Fullsky':
            self.fmask = ''
        elif self.wtype in ['Lmask','LmaskN18']:
            self.fmask = dwin + 'COM_Mask_Lensing_2048_R2.00.fits'
        elif self.wtype == 'LmaskDR3':
            self.fmask = dwin + 'COM_Mask_Lensing_2048_R3.00.fits'
        elif self.wtype == 'G60':
            self.fmask = dwin + 'COM_Mask_Lensing_2048_R2.00_G60.fits'
        elif self.wtype == 'G60Lmask':
            self.fmask = dwin + 'Lensing_mask_with_HFI_GAL060.fits'
        else:
            sys.exit('unknown wtype')
        
        # apodized map
        self.famask = self.fmask.replace('.fits','_a'+str(self.ascale)+'deg.fits')
        if self.wtype == 'LmaskN18':
            self.famask = self.famask.replace('.fits','_N18.fits')
        
        # set original file
        if self.ascale==0.:  self.famask = self.fmask


    def array(self):

        #multipole
        self.l  = np.linspace(0,self.lmax,self.lmax+1)
        self.kL = self.l*(self.l+1)*.5

        #binned multipole
        self.bp, self.bc = basic.aps.binning(self.bn,[self.olmin,self.olmax],self.binspc)

        #theoretical cl
        self.lcl = np.zeros((5,self.lmax+1)) # TT, TE, EE, BB, PP
        self.lcl[:,2:] = np.loadtxt(self.flcl,unpack=True,usecols=(1,2,3,4,5))[:,:self.lmax-1] 
        self.lcl *= 2.*np.pi / (self.l**2+self.l+1e-30)
        self.lcl[0:1,:] /= constants.Tcmb**2 
        self.cpp = self.lcl[4,:]
        self.ckk = self.lcl[4,:] * (self.l**2+self.l)**2/4.



#----------------
# initial setup
#----------------

def init_analysis(**kwargs):
    # setup parameters, filenames, and arrays
    p = analysis(**kwargs)
    analysis.filename(p)
    analysis.array(p)
    analysis.set_mask_filename(p)
    return p


def set_mask(fmask):

    # read window
    w = hp.fitsfunc.read_map(fmask,verbose=False)
    if hp.pixelfunc.get_nside(w) != 2048:
        sys.exit('nside of window is not 2048')

    # binary mask
    M = w/(w+1e-30)

    # normalization
    wn = np.zeros(5)
    wn[0] = np.mean(M)
    for n in range(1,5):
        wn[n] = np.average(w**n)

    return w, M, wn


def wfac(wtype,ascale):

    if wtype == 'Lmask':
        if ascale == 5.:
            wn = [0.66036942,0.56947037,0.54783488,0.53667619,0.52949797]

    return wn


def make_HMLmask(p,overwrite=False):

    mask0 = hp.fitsfunc.read_map(p.Lmask,verbose=False)
    mask1 = hp.fitsfunc.read_map(p.Cmask,field=2,verbose=False)
    masks = mask0 * mask1
    
    hp.fitsfunc.write_map(p.fmask,masks,overwrite=overwrite)


def tau_spec(lmax,Lc=2000.):
    l = np.linspace(0,lmax,lmax+1)
    return  (1e-3)*4.*np.pi/Lc**2*np.exp(-(l/Lc)**2)


def cl_opt_binning(OL,WL,bp):  # binning of power spectrum
    bn = np.size(bp) - 1
    bc = (bp[1:]+bp[:-1])*.5
    cb = np.zeros(bn)
    for i in range(bn):
        b0 = int(bp[i])
        b1 = int(bp[i+1])
        if i==0: print(b0,b1,OL[b0])
        cl = OL[b0:b1]
        wl = 1./WL[b0:b1]**2
        N0 = np.count_nonzero(wl)
        if N0==b1-b0:
            norm = 1./np.sum(wl)
            cb[i] = norm*np.sum(wl*cl)
        elif N0<b1-b0 and N0>0:
            norm = 1./np.sum(wl[wl!=0])
            cb[i] = norm*np.sum(wl[wl!=0]*cl[wl!=0])
        else:
            cb[i] = 0
    return bc, cb


def load_binned_tt(mb,qobj,rlz):
    
    import binning as bn

    # optimal filter
    al = (np.loadtxt(qobj.f['TT'].al)).T[1]
    vl = al/np.sqrt(qobj.l+1e-30)
    # binned spectra
    mtt, __, stt, ott = bn.binned_spec(mb,qobj.f['TT'].cl,cn=1,doreal=True,opt=True,vl=vl)
    # noise bias
    nb = bn.binning( (np.loadtxt(qobj.f['TT'].n0bs)).T[1], mb, vl=vl )
    rd = np.array( [ (np.loadtxt(qobj.f['TT'].rdn0[i])).T[1] for i in rlz ] )
    rb = bn.binning(rd,mb,vl=vl)
    # debias
    ott = ott - rb[0] - nb/(qobj.mfsim)
    mtt = mtt - np.mean(rb[1:,:],axis=0) - nb/(qobj.mfsim-1)
    ott = ott - mtt # subtract average of sim
    stt = stt - rb[1:,:] - nb/(qobj.mfsim-1)
    vtt = np.std(stt,axis=0)
    return ott, mtt, stt, vtt


