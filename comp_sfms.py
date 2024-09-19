import numpy as np
import math as m
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt
#import seaborn as sns
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy import stats
from scipy.optimize import curve_fit
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams.update({'font.size': 18})

#%%
##-------------------
## read in sample-- combined LSBs, Christie 2024
##-------------------
file0 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/combined.csv')
file0.keep_columns(['LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df = file0.to_pandas()
#df = df[df.Z < 0.05]
df = df[df.LOGMSTAR > -99]
df = df[df.LOGSFRSED > - 99]
df = df[df.LOGMSTARERR > -99]
df = df[df.LOGSFRSEDERR > - 99]
print(df.shape)

#%%
##-------------------
## read in sample-- comparison HSB sample, xCOLDGAS (Saintonge 2017)
##-------------------
file1 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/xCOLDGAS.fits')
file1.keep_columns(['logMstar', 'logSFRSED'])
df_xgas= file1.to_pandas()
print(df_xgas.shape)

#%%
##-------------------
## read in sample-- GLSB Du et al 2023
##-------------------
file6 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/du2023.csv')
file6.keep_columns(['M_star', 'SFR'])
df_glsb = file6.to_pandas()
df_glsb['LOGMSTAR'] = np.log10((10**11)*df_glsb['M_star'])
df_glsb['LOGSFR'] = np.log10(df_glsb['SFR'])
#df = df[df.Z < 0.05]
#df_glsb = df_glsb[df_jun.LOGMSTAR > -99]
#df_glsb = df_glsb[df_glsb.LOGSFRSED > - 99]
print(df_glsb.shape)

#%%
##-------------------
## find the curve function
##-------------------
def curve_func(x, a, b, c):
    return a * x - b * x**2 + c * x**3

x_fit = np.linspace(min(df.LOGMSTAR), max(df.LOGMSTAR), 100)
params, _ = curve_fit(curve_func, df.LOGMSTAR, df.LOGSFRSED)
print(params)


#%%
##-------------------
## plot combined SFMS
##-------------------
vals = np.linspace(8, 12, 100)
plt.scatter(df.LOGMSTAR, df.LOGSFRSED, c = 'blue', s = 25, label = 'combined LSBs', alpha = 0.45)
plt.scatter(df_xgas.logMstar, df_xgas.logSFRSED, c = 'black', s = 25, alpha = 0.5, label = 'xCOLDGAS Massive Galaxies')
plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'red', s = 25, label = 'gLSBs', alpha = 0.45)

plt.plot(vals, curve_func(vals, *params), c = 'black', label = 'curve fit')
plt.plot(vals, curve_func(vals, *params)+0.4, c = 'grey', linestyle = '--')
plt.plot(vals, curve_func(vals, *params)-0.4, c = 'grey', linestyle = '--')

plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
plt.legend()
plt.show()

#%%
##-------------------
## compute the specific star formation rate sSFR
##-------------------

df['LOGsSFR'] = np.log10((10**df.LOGSFRSED)/(10**df.LOGMSTAR))

## get the sSFR curve fit
p_sSFR, _ = curve_fit(curve_func, df.LOGMSTAR, df.LOGsSFR)
print(p_sSFR)

#%%

plt.scatter(df.LOGMSTAR, df.LOGsSFR, c = 'blue', s = 25, label = 'combined LSBs', alpha = 0.45)
#plt.scatter(df_xgas.logMstar, df_xgas.logSFRSED, c = 'black', s = 25, alpha = 0.5, label = 'xCOLDGAS Massive Galaxies')
plt.plot(vals, curve_func(vals, *p_sSFR), c = 'black', label = 'curve fit')
plt.plot(vals, curve_func(vals, *p_sSFR)+0.4, c = 'grey', linestyle = '--')
plt.plot(vals, curve_func(vals, *p_sSFR)-0.4, c = 'grey', linestyle = '--')

plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSSFR $yr^{-1}$')
plt.legend()
plt.show()

#%%
##-------------------
## calcualte the residuals
##-------------------
def residual(fit, value):
    return fit - value

residuals = residual(curve_func(df.LOGMSTAR, *params), df.LOGSFRSED)
print(residuals)

#%%
plt.scatter(df.LOGMSTAR, residuals, c = 'blue', s = 20, label = 'combined LSB residuals', alpha = 0.5)
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('residuals')
plt.legend(loc = 'lower right')
plt.show()