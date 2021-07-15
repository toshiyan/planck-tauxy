#!/usr/bin/env python
# coding: utf-8

import numpy as np, reionization as reion

# define parameters
cps = {'H0':67.5,'Om':.31,'Ov':.69,'w0':-1.,'wa':0.}
Ob = .0455
bias = 6.
R0 = 5.
#model = 'TH'
model = 'AS'
ln = 100

# define reionization history
if model=='AS': xe = lambda z: reion.xe_asym(z)
if model=='TH': xe = lambda z: reion.xe_sym(z)

# tau
reion.optical_depth(xe,Ob=Ob,**cps)

for alpha in [0.,1.,2.,3.,4.]:
    # tau spectrum
    l, cl = reion.compute_cltt(xe,R0=R0,alpha=alpha,bias=bias,lmin=1,lmax=3000,ln=ln,zn=500,Ob=Ob,evol=True,**cps)
    
    # save
    np.savetxt('../data/other/tau_'+model+'_R'+str(R0)+'_a'+str(alpha)+'.dat',np.array((l,cl)).T)

