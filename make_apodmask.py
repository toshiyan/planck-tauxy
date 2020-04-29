
import numpy as np, healpy as hp

# * variables
ascale   = 1
filename = 'COM_Mask_Lensing_2048_R2.00'
otag     = 'Lmask'
#filename = 'Lensing_mask_with_HMIS'
#otag     = 'HMLmask'
#filename = 'Lensing_mask_with_HFI_GAL060'
#otag     = 'G60Lmask'

# * Read input binary mask
mask_raw = hp.fitsfunc.read_map('../data/plk/old/public/'+filename+'.fits')
distance = hp.fitsfunc.read_map('../data/plk/old/public/'+filename+'_distance.fits')

# * do process
x = (1.-np.cos(distance))/(1.-np.cos(ascale*np.pi/180.))
x[x>=1.] = 1.
y = np.sqrt(x)
f = y - np.sin(2*np.pi*y)/(2*np.pi)
apomask = f*mask_raw
#apomask = hp.pixelfunc.ud_grade(apomask,1024)
hp.fitsfunc.write_map('../data/plk/old/public/mask_'+otag+str(ascale)+'.fits',apomask,overwrite=True)

