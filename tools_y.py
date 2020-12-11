# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle
import tqdm

# from cmblensplus/wrap
import curvedsky
import basic

# from cmblensplus/utils
import quad_func
import misctools
import cmb as CMB

# from pylib
import planck_filename as plf

# local
import prjlib
import tools_qrec


class compy():  # compton y

    def __init__(self,ytype='milca',masktype=0,ascale=1.0,tausig=False):

        conf = misctools.load_config('COMPY')

        # compsep type
        self.ytype = conf.get('ytype',ytype)

        # y mask
        self.masktype = conf.getint('masktype',masktype)
        self.ymask    = 'M'+str(self.masktype+1)
        self.ascale   = ascale

        # add tauxy signal sim
        self.tausig = tausig

        # tag
        self.ytag = self.ytype+'_'+self.ymask+'_a'+str(self.ascale)+'deg'


    def filename(self,ids):

        d = prjlib.data_directory()

        #//// public data ////#
        # original mask file for ymap
        self.fmask_org = plf.subd['pr2']['ysz'] + 'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits'

        # ymap
        self.fymap  = plf.subd['pr2']['ysz'] + self.ytype + '_ymaps.fits'

        #//// reduced data ////#
        # reduced mask (multiplied ptsr mask and costheta mask)
        self.fmask  = d['win'] + 'ymask.fits'
        self.famask = d['win'] + 'ymask_a'+str(self.ascale)+'deg.fits'
        if self.ascale == 0.0:  self.famask = self.fmask

        # yalm
        ttag = ''
        if self.tausig: ttag = '_tausig'
        self.fyalm  = [d['ysz'] + 'alm/'+self.ytype+'_'+self.ymask+ttag+'_'+x+'.pkl' for x in ids]

        # real clyy
        self.fclyy  = d['ysz'] + 'aps/'+self.ytype+'_'+self.ymask+'.dat'



class xspec():

    def __init__(self,qtag,ytag,ids):

        d = prjlib.data_directory()
        xaps = d['xcor'] + '/aps/'
        
        xtag = qtag + '_' + ytag

        self.xl   = [xaps+'/rlz/xl_'+xtag+'_'+x+'.dat' for x in ids]
        self.xcov = xaps+'/xcov_'+xtag+'.dat'



def init_compy(ids,**kwargs):

    d = prjlib.data_directory()
    
    cy = compy(**kwargs)
    compy.filename(cy,ids)
    
    return cy



def init_cross(qobj,cy,ids,stag,q='TT',est='bh-lens-src'):

    ltag = '_l'+str(qobj.rlmin)+'-'+str(qobj.rlmax)
    xobj = xspec( '_'.join(filter(None,[q,qobj.qtype,qobj.bhe_tag[1:],stag])) + ltag,cy.ytag, ids )
    
    return xobj



def ymap2yalm(cy,Wy,rlz,lmax,w2,ftalm,**kwargs_ov):

    nside = hp.pixelfunc.get_nside(Wy)
    l = np.linspace(0,lmax,lmax+1)
    bl = CMB.beam(10.,lmax)

    for i in tqdm.tqdm(rlz,ncols=100,desc='ymap2alm ('+cy.ytype+','+str(cy.ymask)+')'):

        if misctools.check_path(cy.fyalm[i],**kwargs_ov): continue

        yalm = {}
        clyy = {}

        for yn in [0,1,2]: #full, first, last
        
            if i==0: # real data

                ymap = Wy * hp.fitsfunc.read_map(cy.fymap,field=yn,verbose=False)
                yalm[yn] = curvedsky.utils.hp_map2alm(nside,lmax,lmax,ymap) * bl[:,None]
                clyy[yn] = curvedsky.utils.alm2cl(lmax,yalm[yn])/w2

            else: # generate sim

                clyy[yn] = np.loadtxt(cy.fclyy,unpack=True)[yn+1]
                if ftalm is None:
                    galm = curvedsky.utils.gauss1alm(lmax,clyy[yn][:lmax+1])
                else:
                    talm = pickle.load(open(ftalm[i],"rb"))
                    cltt = prjlib.tau_spec(lmax)
                    clty = np.sqrt(cltt*clyy[yn])*.9
                    __, galm = curvedsky.utils.gauss2alm(lmax,cltt,clyy[yn],clty,flm=talm)
                ymap = Wy * curvedsky.utils.hp_alm2map(nside,lmax,lmax,galm)
                yalm[yn] = curvedsky.utils.hp_map2alm(nside,lmax,lmax,ymap)

        if i == 0:
            xlyy = curvedsky.utils.alm2cl(lmax,yalm[1],yalm[2])/w2
            np.savetxt(cy.fclyy,np.array((l,clyy[0],clyy[1],clyy[2],xlyy)).T)

        pickle.dump((yalm[0],yalm[1],yalm[2]),open(cy.fyalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def quadxy(cy,lmax,rlz,qobj,fx,w3,w2,**kwargs_ov):

    l = np.linspace(0,lmax,lmax+1)

    for i in tqdm.tqdm(rlz,ncols=100,desc='quad x y ('+qobj.qtype+')'):

        if misctools.check_path(fx.xl[i],**kwargs_ov): continue

        # load tau
        tlm = pickle.load(open(qobj.f['TT'].alm[i],"rb"))[0][:lmax+1,:lmax+1]
        mf  = pickle.load(open(qobj.f['TT'].mfb[i],"rb"))[0][:lmax+1,:lmax+1]
        tlm -= mf
        if qobj.qtype=='tau':  tlm = -tlm  
            
        # load yalm
        ylm = {}
        ylm[0], ylm[1], ylm[2] = pickle.load(open(cy.fyalm[i],"rb"))

        # cross spectra
        xl = np.zeros((3,lmax+1))
        for yn in range(3):
            xl[yn,:] = curvedsky.utils.alm2cl(lmax,tlm,ylm[yn][:lmax+1,:lmax+1])/w3

        np.savetxt(fx.xl[i],np.concatenate((l[None,:],xl)).T)



def theta_mask(nside,theta):

    mask = -curvedsky.utils.cosin_healpix(nside)
    v    = np.cos((90.-theta)*np.pi/180.)
    mask[mask>=-v] = 1.
    mask[mask<=-v] = 0.
    
    return mask



def interface(run=['yalm','tauxy'],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},kwargs_y={}):
    
    p = prjlib.init_analysis(**kwargs_cmb)
    W, M, wn = prjlib.set_mask(p.famask)

    # load obscls
    if p.fltr == 'none':
        Wt = W

    if p.fltr == 'cinv':
        Wt = M

    # define objects
    qtau, qlen, qsrc, qtbh, qtBH = tools_qrec.init_quad(p.ids,p.stag,**kwargs_qrec)

    cy = init_compy(p.ids,**kwargs_y)
        
    Wy = hp.fitsfunc.read_map(cy.famask,field=cy.masktype,verbose=False)
    w2 = np.average(Wy**2)
    w3 = np.average(Wy*Wt**2)
    
    if 'yalm' in run:
        ymap2yalm(cy,Wy,p.rlz,p.lmax,w2,p.ftalm,**kwargs_ov)

    if 'tauxy' in run:
        fxtau = init_cross(qtau,cy,p.ids,p.stag,est='')
        quadxy(cy,qtau.olmax,p.rlz,qtau,fxtau,w3,w2,**kwargs_ov)

    if 'tbhxy' in run:
        fxtbh = init_cross(qtbh,cy,p.ids,p.stag,est='bh-lens')
        quadxy(cy,qtbh.olmax,p.rlz,qtbh,fxtbh,w3,w2,**kwargs_ov)

    if 'tBHxy' in run:
        fxtBH = init_cross(qtBH,cy,p.ids,p.stag,est='bh-lens-src')
        quadxy(cy,qtBH.olmax,p.rlz,qtBH,fxtBH,w3,w2,**kwargs_ov)


