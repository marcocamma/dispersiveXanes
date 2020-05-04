import sys
sys.path.insert(0,"../../../")
import collections
import os
import numpy as np
import matplotlib.pyplot as plt

import dispersiveXanes_utils as utils
import xppl37_spectra
import xanes_analyzeRun
import trx
import datastorage as ds


nice_colors = ["#1b9e77", "#d95f02", "#7570b3"]
nice_colors = "#1f78b4 #a6cee3 #b2df8a #33a02c".split()
gradual_colors = ['#014636', '#016c59', '#02818a', '#3690c0', '#67a9cf', '#a6bddb', '#d0d1e6']#, '#ece2f0']

def get_data(run,threshold=0.02,force=False):
  if run == 80:
    refCalibs=slice(1,None,2)
  elif run == 84:
    refCalibs=slice(None,None,2)
  elif run == 60:
    refCalibs=slice(None,None,2)
  elif run == 76:
    refCalibs=slice(None,None)
  else:
    refCalibs=slice(None,None,2)
  fname = "../data/fig_fel_modes_run%04d.h5" % run
  if not os.path.isfile(fname) or force:
    r = xanes_analyzeRun.AnalyzeRun(run=run)
    r.load()
    E = r.E
    calibs = list(r.results.keys())
    calibs.sort()
    calibs = calibs[refCalibs]
    p1 = np.vstack( [r.results[c].p1 for c in calibs] )
    p2 = np.vstack( [r.results[c].p2 for c in calibs] )
    #temp = ds.DataStorage( E=E,p1=p1.astype(np.float16),p2=p2.astype(np.float16))
    temp = ds.DataStorage( E=E,p1=p1,p2=p2)
    _,_,Abs = xppl37_spectra.calcAbs( temp, threshold=0.02 )
    temp.Abs = Abs #.astype(np.float16)
    temp.info="Abs calculated with threshold = 0.02"
    temp.save(fname)
  data = ds.read(fname)
  # nan is saved as -1 for masked arrays
  for k in data.keys():
    try:
      data[k][data[k]==-1] =np.nan
    except (TypeError,AttributeError):
      pass
  print("Run %d → nshots = %d"%(run,len(data.p1)))
  p1,p2,Abs = xppl37_spectra.calcAbs( data, threshold=threshold )
  data.Abs = Abs
  return data
    

def fig_fel_modes(shots_per_run = slice(10,13),showAv=True,force=False,threshold=0.02,smootWidth=0.3):
#  runs = [28,39,54,76,80,84]
#  runs = [80,76,84]
#  runs = [76,80,84]
  runs = [60,76,80,84]
  labels = ['(A)','(B)', '(C)', '(D)']
#  runs = [28,54,76,80,84]
#  figsize = [6,8]
  figsize = [6,8]
  fig,axes = plt.subplots( len(runs),2 , sharex=True, sharey=True,figsize=figsize)
#  fig,axes = plt.subplots( 2,len(runs) , sharex=True, sharey='row')
  #axes = axes.T
  for run,ax,label in zip(runs,axes, labels):
    data = get_data(run,threshold=threshold,force=force)
    E = data.E
    s2 = data.p2[shots_per_run]
    s1 = data.p1[shots_per_run]
    Abs = data.Abs[shots_per_run]
    norm = s2.max()*1.1
    if showAv: ax[0].fill_between(E,0,data.p2.mean(0)/norm,color='#d95f0e',alpha=0.4)
    for ispectrum,(spectrum1,spectrum2,a) in enumerate(zip(s1,s2,Abs)):
      c = nice_colors[ispectrum]
      ax[0].axhline(ispectrum,ls='--',lw=0.5,color=c)
#      ax[1].axhline(ispectrum,ls='--',lw=0.5,color=c)
      ax[0].plot(E,spectrum1/norm+ispectrum,lw=2,color=c)
      ax[1].axhline(0.25+ispectrum,ls='--',lw=1,color=c)
      # smooth does not work with nan's...
      #if smootWidth > 0:
      #a = xppl37_spectra.smoothSpectra(E,a,res=smootWidth)[0]

      ax[1].axvline(7084,lw=0.5,ls='--',color='k')
      ax[1].axvline(7162,lw=0.5,ls='--',color='k')
      ax[1].plot(E,a+0.25+ispectrum,lw=2,color=c)
      noise = np.nanstd(a[200:800]) # calculkate the noise within a spectra rangel
      print(E[200], E[800])
#      noise = np.nanstd(a[400:500])
      ax[1].text(7090,0.5+ispectrum,"σ = %.2f"%noise)
      ax[0].text(7072, 3, "%s"%label, fontsize=12, fontweight='bold', va='top')
#      print(label, run)
#      ax[0].set_ylabel("Spectrum (a.u.) (run %d)"%run, fontsize=9)
      ax[0].set_ylabel("Spectrum (a.u.)", fontsize=10)
      ax[1].set_ylabel("Absorption (a.u.)", fontsize=10)
    ax[0].grid(color="0.8",lw=0.5)
    ax[1].grid(axis='x',color="0.8",lw=0.5)
    tosave = np.vstack( (data.p2.mean(0)/norm,s2/norm) )
#    trx.utils.saveTxt("../data/fig_fel_modes_run%04d_spectra.txt"%run,E,tosave,info="E average_spectrum spectra")
#   trx.utils.saveTxt("../data/fig_fel_modes_run%04d_abs.txt"%run,E,data.Abs[shots_per_run],info="# threshold = %.2f\n# E Abs"%threshold)

  axes[0,0].set_yticks(())
  ax[0].set_xlim(7070,7180)
  ax[0].set_ylim(-0.2,3.2)
  axes[-1,0].set_xlabel("Energy (eV)")
  axes[-1,1].set_xlabel("Energy (eV)")
#  plt.subplots_adjust(left=0.07,right=0.95)
  plt.subplots_adjust(wspace=0.3, hspace=0.1)
#  plt.tight_layout()
  plt.savefig("fig_fel_modes.png",transparent=True,dpi=300) 
  plt.savefig("fig_fel_modes.pdf",transparent=True)

if __name__ == "__main__": fig_fel_modes()
 
