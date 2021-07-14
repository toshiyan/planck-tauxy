#!/usr/bin/env python
# coding: utf-8

import basic, numpy as np, camb, reionization as reion
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# define parameters
cps = {'H0':67.5,'Om':.31,'Ov':.69,'w0':-1.,'wa':0.}
Ob = .0455
bias = 6.
R0 = 5.
model = 'TH'
#model = 'AS'
alpha = 0.
alpha = 2.
ln = 100

# define reionization history
if model=='AS': xe = lambda z: reion.xe_asym(z)
if model=='TH': xe = lambda z: reion.xe_sym(z)

# tau
reion.optical_depth(xe,Ob=Ob,**cps)

# tau spectrum
l, cl = reion.compute_cltt(xe,R0=R0,alpha=alpha,bias=bias,lmin=1,lmax=3000,ln=ln,zn=500,Ob=Ob,**cps)

# save
np.savetxt('../data/tau_'+model+'_R'+str(R0)+'_a'+str(alpha)+'.dat',np.array((l,cl)).T)

'''
# each z contributions
zbin = np.arange(5.,20.,1.)
clz = np.zeros((np.size(l)))

for zi, zb in enumerate(zbin):
    
    print(zb)
    
    for i, L in enumerate(tqdm.tqdm(l)):
        k  = lambda z: L/rz(z)
        Pm = lambda z: Dz(z)**2*Pk(k(z))
        P1 = lambda z: xe(z) * (1-xe(z)) * ( Fk(k(z),Rz(z)) + Gk(k(z),Rz(z),Pm) )
        P2 = lambda z: ( xHlogxH(xe(z))*bias*Ik(k(z),Rz(z)) - xe(z) )**2 * Pm(z)
        I0 = lambda z: Kz(z)*(P1(z)+P2(z))
        I1 = lambda z: Kz(z)*P1(z)
        clz[i] = I0(zb)

    np.savetxt('../data/other/tt_'+model+'_R'+str(R0)+'_a'+str(alpha)+'_z'+str(zb)+'.dat',np.array((l,clz)).T)
'''
