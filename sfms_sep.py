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
## read in sample-- GSWLC, Salim 2016
##-------------------
file0 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/GSWLC.fits')
file0.keep_columns(['Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_gswlc = file0.to_pandas()
#df = df[df.Z < 0.05]
df_gswlc = df_gswlc[df_gswlc.LOGMSTAR > -99]
df_gswlc = df_gswlc[df_gswlc.LOGSFRSED > - 99]
print(df_gswlc.shape)

#%%
##-------------------
## read in sample-- Du 2019
##-------------------
file1 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/du_gswlc.csv')
file1.keep_columns(['RAdeg', 'DEdeg','Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_du = file1.to_pandas()
#df = df[df.Z < 0.05]
df_du = df_du[df_du.LOGMSTAR > -99]
df_du = df_du[df_du.LOGSFRSED > - 99]
print(df_du.shape)

#%%
##-------------------
## read in sample-- Ragaigne 2003
##-------------------
file2 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/rag03_gswlc.csv')
file2.keep_columns(['RAdeg', 'DEdeg', 'Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_rag = file2.to_pandas()
#df = df[df.Z < 0.05]
df_rag = df_rag[df_rag.LOGMSTAR> -99]
df_rag = df_rag[df_rag.LOGSFRSED > - 99]
print(df_rag.shape)

#%%
##-------------------
## read in sample-- O'Neil 2023
##-------------------
file3 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/oneil2023_gswlc.csv')
file3.keep_columns(['RAdeg', 'DEdeg', 'Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_oneil = file3.to_pandas()
#df = df[df.Z < 0.05]
df_oneil = df_oneil[df_oneil.LOGMSTAR > -99]
df_oneil = df_oneil[df_oneil.LOGSFRSED > - 99]
print(df_oneil.shape)

#%%
##-------------------
## read in sample-- Lei 2018
##-------------------
file4 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/lei_gswlc.csv')
file4.keep_columns(['RAdeg', 'DEdeg', 'Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_lei = file4.to_pandas()
#df = df[df.Z < 0.05]
df_lei = df_lei[df_lei.LOGMSTAR > -99]
df_lei = df_lei[df_lei.LOGSFRSED > - 99]
print(df_lei.shape)

#%%
##-------------------
## read in sample-- He 2020
##-------------------
file5 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/he_gswlc.csv')
file5.keep_columns(['Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_he = file5.to_pandas()
#df = df[df.Z < 0.05]
df_he = df_he[df_he.LOGMSTAR > -99]
df_he = df_he[df_he.LOGSFRSED > - 99]
print(df_he.shape)

#%%
##-------------------
## read in sample-- YOLO 2023
##-------------------
file6 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/yolo_gswlc.csv')
file6.keep_columns(['RAdeg', 'DEdeg', 'Z', 'LOGMSTAR', 'LOGMSTARERR', 'LOGSFRSED', 'LOGSFRSEDERR'])
df_yolo = file6.to_pandas()
#df = df[df.Z < 0.05]
df_yolo = df_yolo[df_yolo.LOGMSTAR > -99]
df_yolo = df_yolo[df_yolo.LOGSFRSED > - 99]
print(df_yolo.shape)

#%%
numbs = np.linspace(0, 5059-1, 5059)
print(numbs)

#%%
rand_sampA = np.random.choice(numbs, 500)
rand_sampB = np.random.choice(numbs, 500)
rand_sampC = np.random.choice(numbs, 500)

yolo_stellA = []
yolo_sfrA= []

yolo_stellB = []
yolo_sfrB = []

yolo_stellC = []
yolo_sfrC = []

#%%
for i in range(0, 500):
    a = int(rand_sampA[i])
    b = int(rand_sampB[i])
    c = int(rand_sampC[i])
    print(a, b, c)
    yolo_stellA.append(df_yolo.LOGMSTAR[a])
    yolo_sfrA.append(df_yolo.LOGSFRSED[a])
    print('B')
    yolo_stellB.append(df_yolo.LOGMSTAR[b])
    yolo_sfrB.append(df_yolo.LOGSFRSED[b])
    print('C')
    yolo_stellC.append(df_yolo.LOGMSTAR[c])
    yolo_sfrC.append(df_yolo.LOGSFRSED[c])


#%%
df_yolo_rA = pd.DataFrame()
df_yolo_rA['LOGMSTAR'] = yolo_stellA
df_yolo_rA['LOGSFRSED'] = yolo_sfrA

df_yolo_rB = pd.DataFrame()
df_yolo_rB['LOGMSTAR'] = yolo_stellB
df_yolo_rB['LOGSFRSED'] = yolo_sfrB

df_yolo_rC = pd.DataFrame()
df_yolo_rC['LOGMSTAR'] = yolo_stellC
df_yolo_rC['LOGSFRSED'] = yolo_sfrC


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
## create combined sample of LSBs
##-------------------

df_comb = pd.concat((df_du, df_rag, df_oneil, df_lei, df_he, df_yolo_rA))
print(df_comb.shape)
df_comb.to_csv('combined.csv')

#%%
##---------------------
## read in combined sample
##---------------------
file8 = Table.read('/Users/hannahchristie/Documents/GitHub/GSWLC_matched/combined.csv')
file8.keep_columns(['LOGMSTAR', 'LOGSFRSED'])
df_comb = file8.to_pandas()
#df = df[df.Z < 0.05]
#df_glsb = df_glsb[df_jun.LOGMSTAR > -99]
#df_glsb = df_glsb[df_glsb.LOGSFRSED > - 99]
print(df_comb.shape)

#%%
##-------------------
## plot distribution of LOGMSTAR
##-------------------
plt.hist(df_comb.LOGMSTAR, bins = 100)
plt.xlabel('LOGMSTAR')
plt.ylabel('Count')
plt.show()

#%%
##-------------------
## plot distribution of LOGSFRSED
##-------------------
plt.hist(df_comb.LOGSFRSED, bins = 100)
plt.xlabel('LOGSFR')
plt.ylabel('Count')
plt.show()


#%%
##-------------------
## find the curve function
##-------------------
def curve_func(x, a, b, c):
    return a * x - b * x**2 + c * x**3

x_fit = np.linspace(min(df_comb.LOGMSTAR), max(df_comb.LOGMSTAR), 100)
params, _ = curve_fit(curve_func, df_comb.LOGMSTAR, df_comb.LOGSFRSED)
print(params)
#%%
##-------------------
## find the linear regression parameters
##-------------------

slope, intercept, r, p, std_err = stats.linregress(df_comb.LOGMSTAR, df_comb.LOGSFRSED)

slope_g, intercept_g, r, p, std_err = stats.linregress(df_glsb.LOGMSTAR, df_glsb.LOGSFR)

slope_d, intercept_d, r, p, std_err = stats.linregress(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR)

print(slope, intercept)

#%%
##-------------------
## plot combined SFMS
##-------------------
plt.scatter(df_comb.LOGMSTAR, df_comb.LOGSFRSED, c = 'red', s = 10, label = 'combined LSBs C', alpha = 0.5)
plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
plt.scatter(np.log10(2.7*10**7), np.log10(0.002), marker = '*', c = 'teal', s = 50, label = 'LSBG-285')
plt.scatter(np.log10(2.3*10**7), np.log10(0.009), marker = '*', c = 'deeppink', s = 50, label = 'LSBG-750')

plt.plot(x_fit, curve_func(x_fit, *params), c = 'black', label = 'curve fit')
plt.plot(x_fit, curve_func(x_fit, *params)+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, curve_func(x_fit, *params)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
plt.legend()
plt.show()

#%%
##-------------------
## plot combined SFMS-- linear regression
##-------------------
plt.scatter(df_comb.LOGMSTAR, df_comb.LOGSFRSED, c = 'red', s = 10, label = 'combined LSBs C', alpha = 0.5)
plt.scatter(df_glsb.LOGMSTAR, df_glsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'Du 2023 LSBs')
plt.scatter(df_dlsb.LOGMSTAR, df_dlsb.LOGSFR, c = 'black', s = 10, alpha = 0.5, label = 'McGaugh 2017 LSBs')
plt.plot(x_fit, slope*x_fit + intercept, c = 'black', label = 'curve fit')
plt.plot(x_fit, slope*x_fit + intercept+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, slope*x_fit + intercept-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
plt.legend()
plt.show()

#%%
##-------------------
## plot separated SFMS
##-------------------
plt.scatter(df_du.LOGMSTAR, df_du.LOGSFRSED, c = 'red', s = 10, label = '$\mu_0(B) > 22.5~mag~arcsec^{-2}$')
plt.scatter(df_rag.LOGMSTAR, df_rag.LOGSFRSED, c = 'orange', s = 10, label = '$\mu_{avg, 5}(K_s) > 18~mag~arcsec^{-2}$')
plt.scatter(df_oneil.LOGMSTAR, df_oneil.LOGSFRSED, c = 'green', s = 10, label = '$\mu_{avg}(B) > 25~mag~arcsec^{-2}$')
#plt.scatter(df_lei.LOGMSTAR, df_lei.LOGSFRSED, c = 'blue', s = 10, label = 'Lei 2018')
plt.scatter(df_he.LOGMSTAR, df_he.LOGSFRSED, c = 'purple', s = 10, label = '$\mu_0(B) > 22.5~mag~arcsec^{-2}$')
plt.scatter(df_yolo_rA.LOGMSTAR, df_yolo_rA.LOGSFRSED, c = 'deeppink', s = 10, label = '$\mu_0(B) > 22.5~mag~arcsec^{-2}$')
plt.plot(x_fit, curve_func(x_fit, *params), c = 'black', label = 'curve fit')
plt.plot(x_fit, curve_func(x_fit, *params)+0.4, c = 'grey', linestyle = '--')
plt.plot(x_fit, curve_func(x_fit, *params)-0.4, c = 'grey', linestyle = '--')
plt.xlabel('LOGMSTAR $M_\odot$')
plt.ylabel('LOGSFR $M_\odot yr^{-1}$')
plt.legend(loc = 'lower right')
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








