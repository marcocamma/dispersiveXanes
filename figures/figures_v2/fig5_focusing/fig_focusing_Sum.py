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
from trx import utils
import datastorage as ds
import x3py
import matplotlib.patches as mpatches



runs = [127]
Nshot=[1,3,5,10,20,40]


# this function needs to original HDF5 files to access the XFEL energy


nice_colors = ["#1b9e77", "#d95f02", "#7570b3"]
nice_colors = "#1f78b4 #a6cee3 #b2df8a #33a02c".split()
gradual_colors="#fec44f #fe9929 #ec7014 #cc4c02 #8c2d04 #3690c0 #67a9cf #fec44f #fe9929 #ec7014 #cc4c02 #8c2d04 #3690c0 #67a9cf".split()

                  
color_ss = '#08519c'
color_av = '#238b45'
color_av_all = '#d95f0e'                  
                  
def get_data(run=127,threshold=0.05,force=False, ishotref=0, ishotsam=1):
  fname = "../data/fig_focusing_run%04d.h5" % run
  if not os.path.isfile(fname) or force:
    # this functions splits the data based on FOM < 0.5 --> p1 is the ref, p2 is the sample 
    data=xppl37_spectra.calcSpectraForRun(run,force)
    idx = data.run.results[0].fom < 0.5
    E = data.run.E
    ref    = ds.DataStorage(E=E, p1=data.p1[0],p2=data.p2[0] )
    sample = ds.DataStorage(E=E, p1=data.p1[1],p2=data.p2[1] )

    _,_,Abs = xppl37_spectra.calcAbs(ref,ref,threshold=threshold)
    ref.Abs = Abs

    _,_,Abs = xppl37_spectra.calcAbs(ref,sample,threshold=threshold)
    sample.Abs = Abs

    temp = ds.DataStorage(ref=ref,sample=sample)
    temp.info="Abs calculated with threshold = %.3f" % threshold
    temp.save(fname) 
    print('maybe forced to calculate again the Spectra to get idx_fom')
  
  data = ds.read(fname)
  ref = data.ref
  sample = data.sample
  _,_,Abs = xppl37_spectra.calcAbs(ref,ref,threshold=threshold)
  ref.Abs = Abs
  
  _,_,Abs = xppl37_spectra.calcAbs(ref,sample,threshold=threshold)
  sample.Abs = Abs

  data = ds.DataStorage(ref=ref,sample=sample)
  data["threshold"]=threshold
  
  return data,idx

def get_ref():
  E,data=np.loadtxt("../data/Fe_ref.txt",unpack=True)
  return ds.DataStorage(E=E*1e3,data=data/2.05+0.07)

def get_1b():
  E,data=np.loadtxt("../data/Fe_1bunch.txt",unpack=True)
  return ds.DataStorage(E=E*1e3,data=data/2.05+0.07)

def filter_enerbeam(run=127, ishotsam= 1, nshot=20, nshotmin=400, idx_fom=np.array(1000,dtype=bool)):
    # filter with x-ray pulse intensity (mJ) from gas detector (one could average  the two readings
    # in the case of in/out scan, ishot corresponds to the serie of shots for a same mortor position
    # ie. it has to be equal to ishotsam 
    # idx_fom is the array of index from the initially selected spectra with fom criteria
    fname = "/Users/marionharmand/Documents/0_CNRS/Manip/Manip_2016/LCLS_XPP_May2016/Data/xppl3716-r%04d.h5" %run
    d = x3py.Dataset(fname)
#    data_e=data.copy()
    
    # extract the corresponding measurements of XFEL energy
    data_ener = d.gasdet.f_21_ENRC.data[:]
    data_ener_sam = data_ener[~idx_fom] # in case of 1 calib cycle as for focusing data, and fom selection only
    data_ener_ref = data_ener[idx_fom]
 #   data_ener_sam = data_ener[data.sample.p1.shape[0]*ishotsam:data.sample.p1.shape[0]*2*ishotsam] # for in/out scan

    # select nshot from the min and max data
    idx = np.argsort(data_ener_sam)
    idx_min = idx[:nshotmin]
#    idx_min = idx[:nshot]
    idx_minb = np.zeros(len(data_ener_sam), dtype=bool)
    idx_minb[idx_min] = True
    idx_max = idx[-nshot:]
    idx_maxb = np.zeros(len(data_ener_sam), dtype=bool)
    idx_maxb[idx_max] = True
    idx_max2 = idx[-3*nshot:-2*nshot]
    idx_max2b = np.zeros(len(data_ener_sam), dtype=bool)
    idx_max2b[idx_max2] = True

    idx_ref = np.argsort(data_ener_ref)
    idx_ref_max = idx_ref[-nshot:]
    idx_ref_maxb = np.zeros(len(data_ener_ref), dtype=bool)
    idx_ref_maxb[idx_ref_max] = True
   
    data_ener_min = data_ener_sam[idx_minb]
    data_ener_max = data_ener_sam[idx_maxb]
    data_ener_max2 = data_ener_sam[idx_max2b]
    data_ener_ref_max = data_ener_ref[idx_ref_maxb]
    
    Emin = np.nanmean(data_ener_min,0)
    Emax = np.nanmean(data_ener_max,0)
    print("Emin = %.1f"%Emin, "Emax = %.1f"%Emax)

    return data_ener_max, idx_maxb, data_ener_min, idx_minb, data_ener_max2, idx_max2b, data_ener_ref_max, idx_ref_maxb, Emin, Emax
    
   
#    # select the shots with max intensity
#    curs = 0.9
#    idx_max = data_ener_sam > curs * data_ener_sam.max()
#    data_ener_max = data_ener_sam[idx_max]
#    if data_ener_max.size < nshot:
#        curs = curs - 0.1
#        idx_max = data_ener_sam > curs  * data_ener_sam.max()
#        data_ener_max = data_ener_sam[idx_max]
#    # select the shots with min intensity
#    cursm = 0.1
#    idx_min = data_ener_sam < cursm * data_ener_sam.min()
#    data_ener_min = data_ener_sam[idx_min]
#    if data_ener_min.size < nshot:
#        cursm = cursm + 0.1
#        idx_min = data_ener_sam < cursm * data_ener_sam.mean()
#        data_ener_min = data_ener_sam[idx_min] 
#    return data_ener_max, idx_max, data_ener_min, idx_min


def fig_focusing(run=127,nshot=20,nshotmin = 200,force=False,threshold=0.03,smoothWidth=0.4,i0_monitor=0.1):
# function to analyze the focused data of 1 run with sorting in fucntion of FEL energy beam 
# it gives the spectra in function : 
#       f[1] for spectra with min FEL energy
#       f[2] for spectra with max FEL energy
#       f[3] for references on no sample
  print(nshot)
    
  ref_ESRF = get_ref()
  color_ss = '#08519c'
  color_av = '#238b45'
  color_av_all = '#d95f0e'
  shifty = 1
  data,idx_fom = get_data(run, force=True, threshold=threshold)
  filter_ener = filter_enerbeam(run, nshot=nshot+1, nshotmin=nshotmin, idx_fom=idx_fom)
  E = data.ref.E
  
  # filtering of the min and max energy shots
  data.sample.Abs_max = data.sample.Abs[filter_ener[1]]
  data.sample.p1_max = data.sample.p1[filter_ener[1]]
  data.sample.p2_max = data.sample.p2[filter_ener[1]]
  
  data.sample.Abs_max2 = data.sample.Abs[filter_ener[5]]
  data.sample.p1_max2 = data.sample.p1[filter_ener[5]]
  data.sample.p2_max2 = data.sample.p2[filter_ener[5]]

  data.sample.Abs_min = data.sample.Abs[filter_ener[3]]
  data.sample.p1_min = data.sample.p1[filter_ener[3]]
  data.sample.p2_min = data.sample.p2[filter_ener[3]]
  
  data.ref.Abs_max = data.ref.Abs[filter_ener[7]]
  data.ref.p1_max = data.ref.p1[filter_ener[7]]
  data.ref.p2_max = data.ref.p2[filter_ener[7]]
  
  # extract Absorption spectra
  shots = range(0,min(data.sample.Abs_min.shape[0], data.sample.Abs_max.shape[0]))


  if i0_monitor is not None:
    i0_ref = np.nanmean(data.ref.p1,axis=1)
    idx    = i0_ref>np.percentile(i0_ref,i0_monitor)
    data.ref.Abs = data.ref.Abs[idx]
    
    i0_ref = np.nanmean(data.ref.p1_max,axis=1)
    idx    = i0_ref>np.percentile(i0_ref,i0_monitor) 
    data.ref.Abs_max = data.ref.Abs_max[idx]

    i0_ref = np.nanmean(data.sample.p1_min,axis=1)
    idx    = i0_ref>np.percentile(i0_ref,i0_monitor) 
    data.sample.Abs_min = data.sample.Abs_min[idx]

    i0_ref = np.nanmean(data.sample.p1_max,axis=1)
    idx    = i0_ref>np.percentile(i0_ref,i0_monitor) 
    data.sample.Abs_max = data.sample.Abs_max[idx]
    
    i0_ref = np.nanmean(data.sample.p1_max2,axis=1)
    idx    = i0_ref>np.percentile(i0_ref,i0_monitor) 
    data.sample.Abs_max2 = data.sample.Abs_max2[idx]
    
#  ref = data.ref.Abs[shots]
#  sam_min = data.sample.Abs_min[shots]
#  sam_max = data.sample.Abs_max[shots]
    
  ref = data.ref.Abs
  ref_max = data.ref.Abs_max
  sam_min = data.sample.Abs_min
  sam_max = data.sample.Abs_max
  sam_max2 = data.sample.Abs_max2
  if smoothWidth > 0:
    ref = xppl37_spectra.smoothSpectra(E,ref,res=smoothWidth) 
    ref_max = xppl37_spectra.smoothSpectra(E,ref_max,res=smoothWidth) 
    sam_min = xppl37_spectra.smoothSpectra(E,sam_min,res=smoothWidth) 
    sam_max = xppl37_spectra.smoothSpectra(E,sam_max,res=smoothWidth) 
    sam_max2 = xppl37_spectra.smoothSpectra(E,sam_max2,res=smoothWidth) 
  idx = E>7150 
  sam_min[:,idx]=np.nan
  sam_max[:,idx]=np.nan
  sam_max2[:,idx]=np.nan
  ref[:,idx]=np.nan
  ref_max[:,idx]=np.nan
  av_ref = np.nanmedian(ref,0)
  av_ref_max = np.nanmedian(ref_max,0)
  av_sam_min = np.nanmedian(sam_min,0)
  av_sam_max = np.nanmedian(sam_max,0)
  av_sam_max2 = np.nanmedian(sam_max2,0)
  Nmin = len(sam_min)
  Nmax = len(sam_max)
  
  
  # to save the extracted data
#  to_save = np.vstack( (E,np.nanmedian(ref,0),ref) )
#  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
#  info += "\n#E abs_average_over_shots shots ..."
#  trx.utils.saveTxt("../data/fig_focusing_run%04d_ref.txt"%run,E,to_save,info=info)
#
#  to_save = np.vstack( (E,np.nanmedian(sam_min,0),sam_min) )
#  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
#  info += "\n#E abs_average_over_shots shots ..."
#  trx.utils.saveTxt("../data/fig_focusing_run%04d_sam_min.txt"%run,E,to_save,info=info)
#
#  to_save = np.vstack( (E,np.nanmedian(sam_max,0),sam_max) )
#  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
#  info += "\n#E abs_average_over_shots shots ..."
#  trx.utils.saveTxt("../data/fig_focusing_run%04d_sam_max.txt"%run,E,to_save,info=info)
#  
#  to_save = np.vstack( (E,av_sam_min) )
#  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
#  info += "\n#E abs_average_over_shots shots ..."
#  trx.utils.saveTxt("../data/fig_focusing_run%04d_av_sam_min.txt"%run,E,to_save,info=info)
#
#  to_save = np.vstack( (E,av_sam_max) )
#  info = "# threshold=%.2f; smoothWidth=%.2f eV" %(threshold,smoothWidth)
#  info += "\n#E abs_average_over_shots shots ..."
#  trx.utils.saveTxt("../data/fig_focusing_run%04d_av_sam_max.txt"%run,E,to_save,info=info)
#  
  
  return E, av_sam_min, av_sam_max, av_ref, filter_ener, Nmax, Nmin, av_sam_max2, av_ref_max, nshotmin


def spec_norm(ener,spec,Emin=7095,Emax=7133):
# Normalization of the absorption spectra using 2 points : 
# E0 for the absorption before the edge
# E1 for the absorption after the edge
# spec is obtained from averaging single shots
  
#  spec_sm = spec
#  if smoothWidth > 0:
#    spec_sm = xppl37_spectra.smoothSpectra(E,spec,res=smoothWidth) 

  # to extract the averaged absorption before the edge at E0   
  intmin = spec.copy()
  idxmin = (Emin - 1) > ener 
  intmin[idxmin] = np.nan
  idxmin = ener > (Emin + 1)
  intmin[idxmin] = np.nan
  mumin = np.nanmean(intmin)
 
  # to extract the averaged absorption after the edge at the normalization point E1
  intmax = spec.copy() - mumin
  idxmax = (Emax - 1) > ener 
  intmax[idxmax] = np.nan
  idxmax = ener > (Emax + 1)
  intmax[idxmax] = np.nan
  mumax = np.nanmean(intmax)

# to normalize the spectra
  specnorm = (spec.copy() - mumin) / mumax
   
  return specnorm

    
def fig_focusing_sum(run=127,Nshot=[1,3, 5,10,20],shift=0.3,force=False,threshold=0.02,smoothWidth=0.3,i0_monitor=0.1):
#  color_ss = '#08519c'
#  color_av = '#238b45'
#  color_av_all = '#d95f0e'
    
  ref_ESRF = get_ref()
#  ref_ESRF = get_1b()
   
  figure = plt.figure(figsize = [7,5])
  gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1],)
  ax = []
  ax.append( plt.subplot(gs[0]) )
  ax.append( plt.subplot(gs[1],sharex=ax[0],sharey=ax[0]) ) 

  result = []
  
#  ax[0].set_yticks( () )
#  ax[1].set_yticks( () 

  for idx, ishot in enumerate(Nshot):
      f = fig_focusing(run, nshot = ishot, threshold=threshold)
#      gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
#      ax = []
#      ax.append( plt.subplot(gs[0]) )
#      ax.append( plt.subplot(gs[1],sharex=ax[0],sharey=ax[0]) )
  
      av_ref = f[3]
      av_ref_max = f[8]
      
      filter_ener = f[4]
      Emin = filter_ener[8]
      Emax = filter_ener[9]
      nshotmin = f[9]
      

      Nmax = f[5]
      Nmin = f[6]
      print("run= %.f"%run, "nshot= %.f"%ishot, "Nmax= %.f"%Nmax, "Nmin= %.1f"%Nmin, "Emin = %.1f"%Emin, "Emax = %.1f"%Emax)

    # correction calibration using the reference spectra of ESRF
      E = f[0]
      #Ecal = E 
      #Ecal = (E*(7129+5)/7129)+1.5
      Ecal = 7112 +2 + (E-7112)*1.11
      result.append(Ecal)


    # Normalisation
      av_sam_min = spec_norm(Ecal, f[1], 7102, 7137)
      av_sam_max = spec_norm(Ecal, f[2], 7102, 7137)
      av_sam_max2 = spec_norm(Ecal, f[7], 7102, 7137)
      ref_ESRF.data = spec_norm(ref_ESRF.E, ref_ESRF.data, 7095, 7137)
      
      result.append(av_sam_min)
      result.append(av_sam_max)
      result.append(av_sam_max2)
      
#      ax[0].axhline(idx,ls='--',color="0.9")
#      ax[1].axhline(idx,ls='--',color="0.9")
#      plt.tight_layout()
      
#      noise = np.nanstd(av_ref)
#      ax[0].text(7135,idx*shift + shift ,"σ = %.2f"%noise)


#      ax[1].plot(Ecal,av_sam_min,color=gradual_colors[4],lw=1,zorder=10, label= "400shots E=mJ")
      ax[0].plot(Ecal,idx*shift + shift + av_ref_max,color=gradual_colors[idx],lw=1,zorder=10, label = "%.f shots"%nshotmin + " %.1f mJ"%Emax)
      ax[1].plot(Ecal,idx*shift + shift + av_sam_min,color=nice_colors[-4],lw=1,zorder=10)
      ax[1].plot(Ecal,idx*shift + shift + av_sam_max,color=gradual_colors[idx],lw=1,zorder=10, label= "%.f"%ishot + " shots" + " %.1f mJ"%Emax)
#      ax[1].plot(Ecal,idx*shift + shift + av_sam_max2,color=gradual_colors[5],lw=1,zorder=10, label= "%.2f"%ishot + "shots E=mJ")
#      ax[0].text(7110,idx*shift + shift, "nshot= %.f"%ishot + " σ = %.3f"%np.nanstd(av_ref_max[350:600]))
      ax[0].text(7110,idx*shift + shift, "σ = %.3f"%np.nanstd(av_ref_max[350:600]))
#      ax[0].text(7110,irun*shift,"Run = %.f"%run + " CRL = %.f"%crl + " Nmax = %.f"%Nmax + " Nmin = %.f"%Nmin)
#      ax[0].text(7125,ishot+0.2,"σ = %.2f"%np.nanstd(ref[ishot]))

 

#      ax[0].set_title("Run %s"%str(run))

  ax[0].plot(Ecal,av_ref,color=nice_colors[-4],lw=1,zorder=10, label="%.f shots"%nshotmin + " %.1f mJ"%Emin)
  ax[1].plot(Ecal,0*shift + shift + av_sam_min,color=nice_colors[-4],lw=1,zorder=10, label="%.f shots"%nshotmin + " %.1f mJ"%Emin)
  ax[1].plot(ref_ESRF.E, ref_ESRF.data , color=nice_colors[-2],lw=1, zorder=10, label= "ESRF Reference")
  
  ax[0].grid(axis='x',color="0.7",lw=0.5)
  ax[1].grid(axis='x',color="0.7",lw=0.5)
  ax[0].set_yticks( () )
  ax[1].set_yticks( () )
    
  ax[0].set_ylabel("No sample Absorption (a.u.)")
  ax[1].set_ylabel("Sample Absorption (a.u.)", )
  ax[1].yaxis.tick_right()
  ax[1].yaxis.set_label_position("left")
  ax[0].set_xlabel("Energy (eV)")
  ax[1].set_xlabel("Energy (eV)")
  ax[0].grid(axis='x',color="0.7",lw=0.5)
  ax[1].grid(axis='x',color="0.7",lw=0.5)
  ax[0].set_xlim(7105,7138)
  ax[0].set_ylim(-0.1,2.7) 
  ax[1].legend(loc=0, ncol=1, prop={'size':8})  
    
  # Save figure
  plt.savefig("fig_focusing_NShots.png",transparent=True,dpi=300) 
  plt.savefig("fig_focusing_NShots.pdf",transparent=True)
      
#  return ref_ESRF, E, av_sam_min, av_sam_max
#  return result

if __name__ == "__main__":
    fig_focusing_sum(run=127,Nshot=[1,3, 5,10,20],shift=0.3,force=False,threshold=0.02,smoothWidth=0.3,i0_monitor=0.1)
