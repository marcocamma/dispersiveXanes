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


nice_colors = ["#1b9e77", "#d95f02", "#7570b3"]
nice_colors = "#1f78b4 #a6cee3 #b2df8a #33a02c".split()
gradual_colors = ['#014636', '#016c59', '#02818a', '#3690c0', '#67a9cf', '#a6bddb', '#d0d1e6']#, '#ece2f0']
gradual_colors="#fec44f #fe9929 #ec7014 #cc4c02 #8c2d04 #3690c0 #67a9cf #fec44f #fe9929 #ec7014 #cc4c02 #8c2d04 #3690c0 #67a9cf".split()

def get_data(runs=(155,156),threshold=0.02,force=False):
  run_hash = "_".join(map(str,runs))
  fname = "../data/fig_fe_xas_runs_%s.h5" % run_hash
  if not os.path.isfile(fname) or force:
    E,p1,p2,Abs=xppl37_spectra.calcAbsForRun(runs,merge_calibs=True,threshold=threshold)
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
  # for runs 155,156 the vernier stopped working after shots ~ 2000
  if runs == (155,156):
    data.p1=data.p1[:2000]
    data.p2=data.p2[:2000]
    data.Abs=data.Abs[:2000]
  return data


def get_ref():
  E,data=np.loadtxt("../data/Fe_ref.txt",unpack=True)
  return ds.DataStorage(E=E*1e3,data=data/2.05+0.07)

def get_1b():
  E,data=np.loadtxt("../data/Fe_1bunch.txt",unpack=True)
  return ds.DataStorage(E=E*1e3,data=data/2.05+0.07)

def fig_fe_exas(run=(155,156),first=3,period=20,nSpectra=8,force=False,threshold=0.01,smoothWidth=1.0,i0_filter=0.1):
  ref = get_ref()
  color_ss = '#08519c'
  color_av = '#238b45'
  color_av_all = '#d95f0e'
  shifty = 1
  data = get_data(run,force=force)
  E = data.E;p1=data.p1;p2=data.p2;Abs=data.Abs
  E = (E-7100)*1.938+7133
  p1_sum = p1.sum(-1)
  if i0_filter is not None:
    m = np.percentile(p1_sum,i0_filter*100)
    idx = p1_sum>m
    p1=p1[idx]
    p2=p2[idx]
    Abs=Abs[idx]
    print(idx.sum(),idx.shape[0])
  p1_av  = np.nanmean(p1,axis=0)
  p2_av  = np.nanmean(p2,axis=0)
  shots = slice(first,None,period)


  p1 = p1[shots]; p2=p2[shots]; Abs = Abs[shots]
  if smoothWidth > 0:
    Abs = xppl37_spectra.smoothSpectra(E,Abs,res=smoothWidth)

  figure = plt.figure(figsize = [8,5])
  gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5],wspace=0.15,right=0.97,left=0.05)
  ax = []
  ax.append( plt.subplot(gs[0]) )
  ax.append( plt.subplot(gs[1],sharex=ax[0],sharey=ax[0]) )
  ax.append( plt.subplot(gs[2],sharex=ax[0]) )
  
  ref = get_1b()
  rr = mc.interpolate(ref.E,ref.data,E)

  ref = get_ref()
  rr = mc.interpolate(ref.E,ref.data,E)


  #fig,ax = plt.subplots(1,3,sharex=True,sharey=True,squeeze=False,figsize=[6,8])
  #ax = ax[0]
  normalization = np.nanmax( p1[:nSpectra] )
  to_save = []
  ref = get_ref()
  for ishot,(s1,s2,a) in enumerate(zip(p1[:nSpectra],p2[:nSpectra],Abs[:nSpectra])):
    s1_norm = s1/normalization
    s2_norm = s2/normalization
    offset = (nSpectra-1-ishot)*shifty
    ax[0].axhline(ishot*shifty,ls='--',color="0.9")
    ax[1].axhline(ishot*shifty,ls='--',color=color_ss)
#    ax[1].plot(ref.E,ref.data+(nSpectra-1)*shifty,color=nice_colors[-2],lw=2,zorder=100)
    ax[1].plot(E,np.nanmedian(Abs[:nSpectra],0) + offset,color=color_av_all,lw=2,zorder=10,alpha=0.8)
    ax[0].plot(E,s1_norm + offset,ls = '-' ,color='0.8',lw=2)
    ax[0].plot(E,s2_norm + offset,ls = '-' ,color='0.3',lw=2)
    ax[1].plot(E, a + offset,color=color_ss,lw=2)
    to_save.append(s1_norm)
    to_save.append(s2_norm)
    to_save.append(a)
  nmax = int(np.floor(np.log(len(Abs))/np.log(2)))+1
  print(nmax)
  n = 2**np.arange(nmax)
  print(n)
  n = np.delete(n,0)
 
  ax[2].plot(E,np.nanmedian(Abs[:1],axis=0)+(len(n)+1)*0.2,color=gradual_colors[0],label = "1 shot")
  for i,ni in enumerate(n):
      Abs2 = Abs.copy()
      Abs2 = np.delete(Abs2,i,0) 
      ax[2].plot(E,np.nanmedian(Abs2[:ni],axis=0)+(len(n)-i)*0.2,color=gradual_colors[i+1],label = "%d shots"%ni)
      to_save.insert(i,np.nanmedian(Abs[:ni],axis=0))
  ax[2].plot(ref.E,ref.data,color=nice_colors[-2],lw=2,zorder=100,label="ref ESRF")
#  ax[2].legend(loc=1, ncol=2, prop={'size':8})
#  ax[0].set_title("Run %s"%str(run))
  ax[1].set_ylabel("Sample Absorption (a.u.)")
  ax[2].set_ylabel("Accumulated Sample Absorption(a.u.)")
  ax[0].set_ylabel("Normalized Spectra(a.u.)")
  to_save.insert(i+1,np.nanmedian(Abs[:nSpectra],axis=0))

  to_save.insert(0,rr)
  to_save.insert(1,rr)

  to_save = np.vstack(to_save)
  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
  info += "\n#E esrf_1b esrf_ref "+" ".join(["av_%d_shots"%ni for ni in n]) +" av_%d_shots " % nSpectra + "+ nshots x (spectro1 spectro2 abs)"
  run_hash = "_".join(map(str,run))
 # trx.utils.saveTxt("../data/fig_fe_exas_spectra_runs_%s.txt"%run_hash,E,to_save,info=info)
  ax[0].set_xlabel("Energy (eV)")
  ax[1].set_xlabel("Energy (eV)")
  ax[2].set_xlabel("Energy (eV)")
  ax[0].grid(axis='x',color="0.7",lw=0.5)
  ax[1].grid(axis='x',color="0.7",lw=0.5)
  ax[2].grid(color="0.7",lw=0.5)
  ax[0].set_xlim(7090,7300)
  ax[0].set_yticks( () )
  ax[1].set_yticks( () )
  ax[2].set_yticks( () )
  ax[0].set_ylim(-0.1,nSpectra+0.2)
  ax[1].set_ylim(-0.1,nSpectra+0.2)
  ax[2].set_ylim(0,2.1)
  ax[2].legend(loc=4, ncol=2, prop={'size':8})
  #ax[2].grid(color="0.7",lw=0.5)
  #plt.tight_layout()
  
  plt.savefig("fig_fe_exas.png",transparent=True,dpi=300) 
  plt.savefig("fig_fe_exas.pdf",transparent=True)
  

#if __name__ == "__main__": fig_fe_xas()
 
