# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:31:01 2017
Program to run the whole stuff of Marco
@author: harmand
"""
#To run in the right repertory, do:
#cd /reg/d/psdm/xpp/xppl3716/res/dispersiveXanes
#cd /Users/marionharmand/Documents/0_CNRS/Manip/Manip_2016/LCLS_XPP_May2016/Analysis/dispersiveXanes

import matplotlib
import matplotlib.pylab as plt
matplotlib.style.use("ggplot")
import pprint
import numpy as np
np.warnings.simplefilter('ignore')

import sys; 


############################################################
# Initialization
############################################################


#### PATH - to modify
#### Note: the path in the program xanes_analyseRun might be changed if needed as well to save in the same folder
sys.path.insert(0,"/Users/marionharmand/Documents/0_CNRS/Manip/Manip_2016/LCLS_XPP_May2016/Analysis/dispersiveXanes_FilterEner")
import xanes_analyzeRun
import alignment
plt.ion()


r = xanes_analyzeRun.AnalyzeRun(86,initAlign="/Users/marionharmand/Documents/0_CNRS/Manip/Manip_2016/LCLS_XPP_May2016/Analysis/dispersiveXanes_FilterEner/xppl3716_init_pars/run0084_transform.npy")

# To do the fit and save the transform
r0fit=r.doShot(shot=11,calib=8,showInit=True,doFit=True)


r.doShots(shots=slice(15,20),calib=9,doFit=True)
r.analyzeScan()
#r.save(overwrite=True)
#r.saveTransform()