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
## read in sample-- combined sample of LSBs cross matched (5'') with AFLALFA 2018
##-------------------
file0 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/combxALF.csv')
file0.keep_columns(['Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR', 'logMHI', 'e_logMHI'])
df = file0.to_pandas()
#df = df[df.Z < 0.05]
df = df[df.LOGMSTAR > -99]
df = df[df.LOGSFRSED > - 99]
print(df.shape)

#%%
##-------------------
## read in sample-- combined sample of LSBs cross matched (5'') with xGASS 2018
##-------------------
file1 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/comb_xGASS.csv')
file1.keep_columns(['Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR', 'logMHI'])
dfx = file1.to_pandas()
#df = df[df.Z < 0.05]
dfx = dfx[dfx.LOGMSTAR > -99]
dfx = dfx[dfx.LOGSFRSED > - 99]
print(dfx.shape)

#%%
g_comb = pd.concat((df, dfx))
print(g_comb.shape)
g_comb.to_csv('gas_comb.csv')

#%%
##-------------------
## read in sample-- combined sample of LSBs cross matched (5'') with AFLALFA 2018
##-------------------
file3 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/gas_comb.csv')
file3.keep_columns(['LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR', 'logMHI'])
df_comb = file3.to_pandas()
#df = df[df.Z < 0.05]
df_comb = df_comb[df_comb.LOGMSTAR > -99]
df_comb = df_comb[df_comb.LOGSFRSED > - 99]
print(df_comb.shape)


#%%
##-------------------
## calculate star formation efficiency SFE
##-------------------

df['LOGSFE'] = np.log10((10**df.LOGSFRSED) / (10**df.logMHI))
dfx['LOGSFE'] = np.log10((10**dfx.LOGSFRSED) / (10**dfx.logMHI))
df_comb['LOGSFE'] = np.log10((10**df_comb.LOGSFRSED) / (10**df_comb.logMHI))
print(df_comb.LOGSFE)

#%%
##-------------------
## calculate depletion times, SFR/M_HI
##-------------------

df['t_dep'] = (10**df.logMHI)/(10**df.LOGSFRSED)
dfx['t_dep'] = (10**dfx.logMHI)/(10**dfx.LOGSFRSED)
df_comb['t_dep'] = (10**df_comb.logMHI)/(10**df_comb.LOGSFRSED)
print(df_comb.t_dep)
#%%
##-------------------
## plot depletion times in a histogram
##-------------------

plt.hist(np.log10(df.t_dep), bins=10)
plt.xlabel('LOG$t_{dep}$')
plt.show()

#%%
##-------------------
## plot distribution of BA_LEDA
##-------------------
plt.hist(df.logMHI, bins = 10)
plt.xlabel('LOG(HI Mass)')
plt.ylabel('Count')
plt.show()


#%%
##-------------------
## find the linear regression function
##-------------------
x_fit = np.linspace(min(df_comb.LOGMSTAR), max(df_comb.LOGMSTAR), 100)
slope, intercept, r, p, std_err = stats.linregress(df_comb.LOGMSTAR, df_comb.logMHI)
print(slope, intercept)

#%%
##-------------------
## plot combined SFMS
##-------------------
plt.scatter(df.LOGMSTAR, df.logMHI, c = 'red', s = 25, label = 'combined LSBs')
plt.scatter(dfx.LOGMSTAR, dfx.logMHI, c = 'red', s = 25)

#plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'red', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
#plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'red', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
#plt.scatter(np.log10(2.7*10**7), np.log10(0.002), marker = '*', c = 'teal', s = 50, label = 'LSBG-285')
#plt.scatter(np.log10(2.3*10**7), np.log10(0.009), marker = '*', c = 'deeppink', s = 50, label = 'LSBG-750')

plt.plot(x_fit, slope * x_fit + intercept, c = 'black', label = 'linear regression fit')
plt.plot(x_fit, slope * x_fit + intercept+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, slope * x_fit + intercept-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR ($M_\odot$)')
plt.ylabel('LOGM_HI ($M_\odot$)')
plt.legend()
plt.show()


#%%
##-------------------
## calculate gas fractions, assuming Mgas = MHI
##-------------------
df_comb['fg'] = (10**df_comb.logMHI)/(10**df_comb.LOGMSTAR)
print(df_comb.fg)



#%%
##-------------------
## plot gas fractions, assuming Mgas = MHI, as a function of stellar mass
##-------------------

plt.scatter(df_comb.LOGMSTAR, df_comb.fg, c = 'red', label = 'combined LSBs')
plt.xlabel('LOGMSTAR ($M_\odot$)')
plt.ylabel('Gas Fraction')
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


