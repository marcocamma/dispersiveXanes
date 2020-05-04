import sys
sys.path.insert(0,"../../../")
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import dispersiveXanes_utils as utils
import xppl37_spectra
import xanes_analyzeRun
import mcutils as mc
import trx
import datastorage as ds


# !!!!!DO WITH RUN 84? + add arroys!

#fig_fe_xas(run = 82, shots = [20, 21,33,31,29], smoothWidth = 0.4)
#fig_fe_xas(run = 82, shots = [11,21,33,31,29], smoothWidth = 0.4)

nice_colors = ["#1b9e77", "#d95f02", "#7570b3"]
#nice_colors = "#1f78b4 #a6cee3 #b2df8a #33a02c #FFDEAD".split()
nice_colors = "#1f78b4 #a6cee3 #b2df8a #33a02c".split()
gradual_colors = ['#014636', '#016c59', '#02818a', '#3690c0', '#67a9cf', '#a6bddb', '#d0d1e6']#, '#ece2f0']

def get_data(run=82,calib=1,threshold=0.02,force=False):
  fname = "../data/fig_fe_xas_run%04d.h5" % run
  if not os.path.isfile(fname) or force:
    E,p1,p2,Abs=xppl37_spectra.calcAbsForRun(run,merge_calibs=True,threshold=threshold)
    temp = ds.DataStorage( E=E,p1=p1,p2=p2,Abs=Abs)
    temp.info="Abs calculated with threshold = %.3f" % threshold
    temp.save(fname)
  data = ds.read(fname)
  # nan is saved as -1 for masked arrays
  for k in data.keys():
    try:
      data[k][data[k]==-1] =np.nan
    except TypeError:
      pass
  return data


def get_ref():
  E,data=np.loadtxt("../data/Fe_ref.txt",unpack=True)
  return ds.DataStorage(E=E*1e3,data=data/2.05+0.07)

def get_1b():
  E,data=np.loadtxt("../data/Fe_1bunch.txt",unpack=True)
  return ds.DataStorage(E=E*1e3,data=data/2.05+0.07)

def fig_fe_xas(run=82,shots = slice(111,116),showAv=True,force=False,threshold=0.02,smoothWidth=0.3):
  ref = get_ref()
  color_ss = '#08519c'
  color_av = '#238b45'
  color_av_all = '#d95f0e'
  shifty = 1
  data = get_data(run,force=force)
  E = data.E;p1=data.p1;p2=data.p2;Abs=data.Abs
  p1_sum = p1.sum(-1)
  p1_av  = np.nanmean(p1,axis=0)
  p2_av  = np.nanmean(p2,axis=0)
  # somehow nanmedian screws up when array is too big ... so using nanmean
  abs_av = np.nanmean(Abs,axis=0)
  n = 2**np.arange(4)
  av = np.nanmedian(Abs[:],0)
  av = xppl37_spectra.smoothSpectra(E,av,res=smoothWidth)
  for ni in n:
    aa = np.nanmedian(Abs[:ni],0)
    aa = xppl37_spectra.smoothSpectra(E,aa,res=smoothWidth)
    print(ni,np.nanstd(aa-av))


  p1 = p1[shots]; p2=p2[shots]; Abs = Abs[shots]
  if smoothWidth > 0:
    Abs = xppl37_spectra.smoothSpectra(E,Abs,res=smoothWidth)
    idx = E< 7080
    Abs[:,idx]=np.nan
  #fig,ax = plt.subplots(1,3,sharex=True,sharey=False,squeeze=False,figsize=[6,4])
  figure = plt.figure(figsize = [7,5])
  gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5],)
  ax = []
  ax.append( plt.subplot(gs[0]) )
  ax.append( plt.subplot(gs[1],sharex=ax[0],sharey=ax[0]) )
  ax.append( plt.subplot(gs[2],sharex=ax[0]) )
  
  #ax = ax[0]
  normalization = np.nanmax( p1 )
  to_save = []
  for ishot,(s1,s2,a) in enumerate(zip(p1,p2,Abs)):
    s1_norm = s1/normalization
    s2_norm = s2/normalization
    ax[0].axhline(ishot*shifty,ls='--',color="0.9")
    ax[1].axhline(ishot*shifty,ls='--',color=color_ss)
    if showAv:
      ax[1].plot(E,np.nanmedian(Abs,0)+ishot*shifty,color=color_av_all,lw=1,zorder=10,alpha=0.8)
    ax[0].plot(E,s1_norm+ishot*shifty,ls = '-' ,color='0.8',lw=2)
    ax[0].plot(E,s2_norm+ishot*shifty,ls = '-' ,color='0.3',lw=2)
    ax[1].plot(E,a+ishot*shifty,color=color_ss,lw=2)
    #ax[1].plot(ref.E,ref.data+ishot*shifty,color=color_av_all,lw=2,zorder=100)
    to_save.append(s1_norm)
    to_save.append(s2_norm)
    to_save.append(a)
#  ax[0].set_title("Run %s"%str(run))
  ax[1].set_ylabel("Sample Absorption (a.u.)")
  ax[0].set_ylabel("Normalized Spectra (a.u.)")

  ax[2].errorbar(E,xppl37_spectra.smoothSpectra(E,Abs[2],res=smoothWidth)[0]+0.20, 0.02, alpha=0.1, color=color_ss)
  ax[2].plot(E,xppl37_spectra.smoothSpectra(E,Abs[2],res=smoothWidth)[0]+0.20,color=color_ss,label="1 shot LCLS")
  ax[2].plot(E,xppl37_spectra.smoothSpectra(E,np.nanmedian(Abs,axis=0),res=smoothWidth)[0]+0.15,color=color_av_all,label="5 shots LCLS")
  ref = get_1b()
  ax[2].plot(ref.E,ref.data-0.00,'-x', markevery=[0,-116], color=nice_colors[-4],label="1 shot ESRF")
  ref = get_ref()
  ax[2].plot(ref.E,ref.data-0.05,'-o', markevery=[0,40], color=nice_colors[-2], label="ref ESRF")
#  ax[2].legend()
  handles,labels = ax[2].get_legend_handles_labels()
  handles = [handles[0], handles[1], handles[2], handles[3]]
  labels = [labels[0], labels[1], labels[2],labels[3]]
  ax[2].legend(handles,labels,loc=0)

  ref = get_1b()
  rr = mc.interpolate(ref.E,ref.data,E)
  to_save.insert(0,rr)
  ref = get_ref()
  rr = mc.interpolate(ref.E,ref.data,E)
  to_save.insert(1,rr)
  to_save.insert(2,xppl37_spectra.smoothSpectra(E,np.nanmedian(Abs,axis=0),res=smoothWidth)[0])
 
  to_save = np.vstack(to_save)
  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
  info += "\n#E esrf_1b esrf_ref abs_average_over_shots nshots x (spectro1 spectro2 abs)"
#  trx.utils.saveTxt("../data/fig_fe_xas_spectra_run%04d.txt"%run,E,to_save,info=info)
  ax[0].set_xlabel("Energy (eV)")
  ax[1].set_xlabel("Energy (eV)")
  ax[2].set_xlabel("Energy (eV)")
  ax[0].grid(axis='x',color="0.7",lw=0.5)
  ax[1].grid(axis='x',color="0.7",lw=0.5)
  ax[0].set_xlim(7070,7180)
  ax[1].set_xlim(7070,7180)
  ax[2].set_xlim(7105,7150)
  ax[0].set_yticks( () )
  ax[1].set_yticks( () )
  ax[2].set_yticks( () )
  ax[0].set_ylim(-0.1,len(Abs)+0.2)
  ax[1].set_ylim(-0.1,len(Abs)+0.2)
  ax[2].set_ylim(0,0.9)
  ax[2].grid(color="0.7",lw=0.5)
  plt.tight_layout()
  
  plt.savefig("fig_fe_xas_2.png",transparent=True,dpi=300) 
  plt.savefig("fig_fe_xas_2.pdf",transparent=True)
  

#if __name__ == "__main__": fig_fe_xas()
 
