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
                  
                  
                  
def fig_theta_scan(run=54,shots = slice(111,116),showAv=True,force=False,threshold=0.02,smoothWidth=0.3):
  ref = get_ref()
  color_ss = '#08519c'
  color_av = '#238b45'
  color_av_all = '#d95f0e'
  shifty = 1


 
  r = xanes_analyzeRun.readDataset(run)
  theta = r.scan.spec2th2 
  r2 = xanes_analyzeRun.AnalyzeRun(run, initAlign="dispersiveXanes_FilterEner/xppl3716_init_pars/run00%.f_transform.npy"%run)
  r2.analyzeScan()
#  bestfom = []
#  bestfom_nosam = []   # in case of an in/out scan
  bestfom_all = []
  ratio_all = []
  int2_all = []
  s2norm_all = []

  for c in range(len(theta)):
      ret = r2.results[c]
      bestfom = np.nanmin(ret.fom)
      bestfom_all.append(bestfom)

      int1 = np.sum(ret.p1_sum)
      int2 = np.sum(ret.p2_sum)
      ratio = int2 / int1
      ratio_all.append(ratio)


  ax=plt.gca()
  plt.plot(theta, ratio_all, color=nice_colors[-4], lw=1)
  plt.plot(theta, ratio_all, '.k', markersize=10)
  plt.ylabel("Norm. 2nd spectro. Intensity", fontsize = 16)
  plt.xlabel("2nd Spectro. diffraction angle (degree)", fontsize = 16)
  plt.grid(axis='x',color="0.7",lw=0.5) 
  
  ax.spines['bottom'].set_color('k')
  ax.spines['top'].set_color('k')
  ax.spines['left'].set_color('k')
  ax.spines['right'].set_color('k')




  plt.savefig("fig_theta_scan.png",transparent=True,dpi=300) 
  plt.savefig("fig_theta_scan.pdf",transparent=True)  
  
# if the run is also and in/out scan
#  bestfom_sam = bestfom_all[1::2]
#  bestfom_nosam = bestfom_all[0::2]  
  
  #plt.plot(theta, bestfom_all, '.r')
  
  
  
  