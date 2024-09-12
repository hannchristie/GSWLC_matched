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
## read in sample-- combined samle of LSBs cross matched (5'') with SGA 2020
##-------------------
file0 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/combxSGA.fits')
file0.keep_columns(['Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR', 'D25_LEDA', 'BA_LEDA', "SB_D25_LEDA"])
df = file0.to_pandas()
#df = df[df.Z < 0.05]
df = df[df.LOGMSTAR > -99]
df = df[df.LOGSFRSED > - 99]
print(df.shape)

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
## read in sample-- McGaugh 2017
##-------------------
file7 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/mcgaugh2017.csv')
file7.keep_columns(['LOGMSTAR', 'LOGSFR'])
df_dlsb = file7.to_pandas()
#df = df[df.Z < 0.05]
#df_glsb = df_glsb[df_jun.LOGMSTAR > -99]
#df_glsb = df_glsb[df_glsb.LOGSFRSED > - 99]
print(df_dlsb.shape)


#%%
##-------------------
## plot distribution of D25_LEDA
##-------------------
plt.hist(df.D25_LEDA, bins = 50)
plt.xlabel('D25_LEDA (arcmin)')
plt.ylabel('Count')
plt.show()

#%%
##-------------------
## plot distribution of BA_LEDA
##-------------------
plt.hist(df.BA_LEDA, bins = 50)
plt.xlabel('b/a ratio')
plt.ylabel('Count')
plt.show()


#%%
##-------------------
## find the curve function
##-------------------
def curve_func(x, a, b, c):
    return a * x - b * x**2 + c * x**3
## get curve params from total sample
paramsA = -2.50865828
paramsB = -0.44563662
paramsC = -0.01967605

#%%
##-------------------
## plot combined SFMS
##-------------------
plt.scatter(df.LOGMSTAR, df.LOGSFRSED, c = df.D25_LEDA, cmap = 'viridis', vmin = 0.25, vmax = 1, s = 10, label = 'combined LSBs')
#plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'red', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
#plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'red', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
#plt.scatter(np.log10(2.7*10**7), np.log10(0.002), marker = '*', c = 'teal', s = 50, label = 'LSBG-285')
#plt.scatter(np.log10(2.3*10**7), np.log10(0.009), marker = '*', c = 'deeppink', s = 50, label = 'LSBG-750')

plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC), c = 'black', label = 'curve fit')
plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
cbar = plt.colorbar()
cbar.set_label('D25_LEDA', fontsize = 15)
plt.legend()
plt.show()


#%%
##-------------------
## plot size (D25)- stellar mass relation
##-------------------
plt.scatter(df.LOGMSTAR, df.D25_LEDA, c = df.SB_D25_LEDA, cmap = 'spring', s = 10, label = 'combined LSBs')
#plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
#plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
#plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC), c = 'black', label = 'curve fit')
#plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)+0.4, c = 'grey', linestyle = '--')
#plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('D25_LEDA (arcmin)')
cbar = plt.colorbar()
cbar.set_label('SB D25', fontsize = 15)
plt.legend()
plt.show()

#%%
##-------------------
## plot size (D25)- stellar mass relation
##-------------------
plt.scatter(df.LOGMSTAR, df.D25_LEDA, c = df.SB_D25_LEDA, cmap = 'spring', s = 10, label = 'combined LSBs')
#plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
#plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
#plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC), c = 'black', label = 'curve fit')
#plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)+0.4, c = 'grey', linestyle = '--')
#plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('D25_LEDA (arcmin)')
cbar = plt.colorbar()
cbar.set_label('SB D25', fontsize = 15)
plt.legend()
plt.show()


#%%
##-------------------
## plot combined SFMS with SB as colour
##-------------------
plt.scatter(df.LOGMSTAR, df.LOGSFRSED, c = df.SB_D25_LEDA, cmap = 'spring', s = 10, label = 'combined LSBs')
#plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'red', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
#plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'red', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
#plt.scatter(np.log10(2.7*10**7), np.log10(0.002), marker = '*', c = 'teal', s = 50, label = 'LSBG-285')
#plt.scatter(np.log10(2.3*10**7), np.log10(0.009), marker = '*', c = 'deeppink', s = 50, label = 'LSBG-750')

plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC), c = 'black', label = 'curve fit')
plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, curve_func(x_fit, paramsA, paramsB, paramsC)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
cbar = plt.colorbar()
cbar.set_label('SB D25 $mag~arcsec^{-2}$', fontsize = 15)
plt.legend()
plt.show()
#%%
##-------------------
## calcualte the residuals
##-------------------
def residual(fit, value):
    return fit - value

residuals = residual(curve_func(df_comb.LOGMSTAR, *params), df_comb.LOGSFRSED)
print(residuals)

#%%
plt.scatter(df_comb.LOGMSTAR, residuals, c = 'black', s = 10, label = 'combined LSB residuals')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('residuals')
plt.legend(loc = 'lower right')
plt.show()

#%%

file7 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/comb_col.csv')
file7.keep_columns(['RAdeg', 'DEdeg', 'LOGMSTAR', 'LOGSFRSED', 'gmag', 'rmag', 'imag'])
df_col = file7.to_pandas()
df_col = df_col[df_col.LOGMSTAR > -99]
df_col = df_col[df_col.LOGSFRSED > -99]

print(max(df_col.gmag - df_col.imag))
#%%
##-------------------
## plot combined SFMS with colour from SDSS
##-------------------
plt.scatter(df_col.LOGMSTAR, df_col.LOGSFRSED, c = (df_col.gmag - df_col.imag), cmap = 'RdBu_r', vmin =  0.5, vmax = 1.5, s = 10, label = 'combined LSBs A')
#plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
#plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')

#plt.scatter(df_jun.LOGMSTAR, df_jun.LOGSFRSED, c = 'grey', s = 5, alpha = 0.5, label = 'Junais LSBs')
plt.plot(x_fit, curve_func(x_fit, *params), c = 'red', label = 'curve fit')
plt.plot(x_fit, curve_func(x_fit, *params)+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, curve_func(x_fit, *params)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
cbar = plt.colorbar()
cbar.set_label('g - i', fontsize = 15)
plt.legend()
plt.show()








