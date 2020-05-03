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


def data_directory():
    
    direct = {}

    root = '/global/cscratch1/sd/toshiyan/plk/'
    direct['dr2']  = '/project/projectdirs/cmb/data/planck2015/'
    direct['dr3']  = '/project/projectdirs/cmb/data/planck2018/'
    direct['root'] = root
    direct['inp']  = root + 'input/'
    direct['win']  = root + 'mask/'
    direct['cmb']  = root + 'cmb/'
    direct['bem']  = root + 'beam/'
    direct['ysz']  = root + 'ysz/'
    direct['xcor'] = root + 'xcor/'
    direct['ptsr'] = root + 'ptsr/'

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


class xbispec():

    def __init__(self,xdir,qtag,ytag,otag,ids):

        xtag = qtag + '_' + ytag
        xaps = xdir + '/bsp/'

        self.mbsp = xaps+'bl_'+xtag+otag+'.dat'
        self.obsp = xaps+'bl_obs_'+xtag+otag+'.dat'
        self.mxsp = xaps+'xl_'+xtag+otag+'.dat'
        self.oxsp = xaps+'xl_obs_'+xtag+otag+'.dat'


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
        if self.dtype!='dr2_nilc':  
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
        self.ids = [str(i).zfill(5) for i in range(-1,200)]
        self.ids[0] = 'real'  # change 1st index
        ids = self.ids

        #//// Input public maps ////#
        # sim
        PLK = d[self.dtype[:3]]
        self.fimap = {}

        # PLANCK DR2
        if self.dtype in ['dr2_nilc','dr2_smicaffp8','dr2_nosz']:
            self.fimap['s']  = [PLK+'ffp8/compsep/mc_cmb/ffp8_'+self.dtype.replace('dr2_','')+'_int_cmb_mc_'+x+'_005a_2048.fits' for x in ids]
            self.fimap['n']  = [PLK+'ffp8/compsep/mc_noise/ffp8_'+self.dtype.replace('dr2_','')+'_int_noise_mc_'+x+'_005a_2048.fits' for x in ids]

        if self.dtype == 'dr2_smica':
            self.fimap['s']  = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_cmb_new_mc_'+x+'_005a_2048.fits' for x in ids]
            self.fimap['n']  = [PLK+'ffp8.1/compsep/dx11_v2_smica_int_noise_mc_'+x+'_005a_2048.fits' for x in ids]
        
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
        
        if self.dtype == 'dr2_nosz':
            self.fimap['s'][0] = d['cmb']+'map/nosz.fits'

        # input klm realizations
        self.fiklm = [ d['inp'] + 'sky_klms/sim_'+x[1:]+'_klm.fits' for x in ids ]
        if self.tausig:
            self.ftalm = [ d['inp'] + 'sky_tlms/sim_tausig_'+x[1:]+'_tlm.fits' for x in ids ]
        else:
            self.ftalm = None

        # PLANCK DR3
        #if self.dtype in ['nilc','smica','nosz']:
        #    self.fimap['s']  = [PLK+'ffp10/compsep/mc_cmb/ffp8_'+self.dtype+'_int_cmb_mc_'+x+'_005a_2048.fits' for x in ids]
        #    self.fimap['n']  = [PLK+'ffp10/compsep/mc_noise/ffp8_'+self.dtype+'_int_noise_mc_'+x+'_005a_2048.fits' for x in ids]

        #//// base best-fit cls ////#
        # aps of Planck 2015 best fit cosmology
        self.flcl = d['inp'] + 'COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt'

        #//// Derived data filenames ////#
        # cmb map, alm and aps
        self.fcmb = cmb(self)

        # beam
        self.fbeam = d['bem'] + self.dtype+'.dat'
        if self.dtype == 'dr2_nosz': 
            self.fbeam = d['bem'] + 'dr2_smica.dat'

        # ptsr seed
        ptag = self.wtype+'_a'+str(self.ascale)+'deg'
        self.fpseed  = [d['ptsr']+x+'.fits' for x in ids]
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
        elif self.wtype == 'Lmask':
            self.fmask = dwin + 'COM_Mask_Lensing_2048_R2.00.fits'
        elif self.wtype == 'G60':
            self.fmask = dwin + 'COM_Mask_Lensing_2048_R2.00_G60.fits'
        elif self.wtype == 'G60Lmask':
            self.fmask = dwin + 'Lensing_mask_with_HFI_GAL060.fits'
        else:
            sys.exit('unknown wtype')
        
        # apodized map
        self.famask = self.fmask.replace('.fits','_a'+str(self.ascale)+'deg.fits')
        
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


def set_mask(fmask,wtype='',verbose=False):

    # read window
    if wtype == 'Fullsky':
        w = 1.
    else:
        w = hp.fitsfunc.read_map(fmask,verbose=verbose)
        if hp.pixelfunc.get_nside(w) != 2048:
            sys.exit('nside of window is not 2048')

    # normalization
    wn = np.zeros(5)
    for n in range(1,5):
        wn[n] = np.average(w**n)

    # binary mask
    M = w/(w+1e-30)
    wn[0] = np.mean(M)

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


