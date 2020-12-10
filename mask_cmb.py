#!/usr/bin/env python

import numpy as np, healpy as hp, curvedsky, misctools, prjlib
from matplotlib.pyplot import *

ascale = 1. # 1deg apodization to Galactic mask
nside = 2048

#for wtype, gal in [('Lmask',3),('G60',2),('G60Lmask',2)]: #index is G70 or G60
for wtype in ['LmaskN18']: #index is G70 or G60
    p = prjlib.init_analysis(wtype=wtype,ascale=ascale)

    if not misctools.check_path(p.famask,overwrite=True):
        mask = hp.fitsfunc.read_map(p.fmask,verbose=True)
        amask = curvedsky.utils.apodize(nside,mask,ascale) # apodize original mask
        hp.fitsfunc.write_map(p.famask,amask,overwrite=True)

    p = prjlib.init_analysis(wtype=wtype,ascale=ascale)
    mask = hp.fitsfunc.read_map(p.fmask)
    print('effective sky area',np.mean(mask))
    amask = hp.fitsfunc.read_map(p.famask,verbose=False)
    print('effective sky area',np.mean(amask))
    