from cosmoprimo.fiducial import AbacusSummit
import scipy as sc
import numpy as np


def compute_growth(cosmo_model):

    #-- Definition of "time" = ln(a)
    a = np.logspace(-2, 0, 10000)
    ln_a = np.log(a)
    z = 1/a -1
    H = cosmo_model.hubble_function(z)
    dH_dlna = -(1+z)*np.gradient(H,z) #np.gradient(H,ln_a) # 
    
    def get_dH_dlna(cosmo_model,ln_a_new):
        return np.interp(ln_a_new,ln_a,dH_dlna)
        
    def df_over_dlna(f, ln_a):
        a = np.exp(ln_a)
        z = 1/a - 1
        H = cosmo_model.hubble_function(z)
        dH = get_dH_dlna(cosmo_model,ln_a)
        
        deriv = -f**2 - f*(2 + dH/H) +3/2*cosmo_model.Omega0_m*a**(-3) * (cosmo_model.H0/H)**2
        return deriv 

    #-- Initial condition 
    f0 = 1.
    f = sc.integrate.odeint(df_over_dlna, f0, ln_a)
    return a, f
