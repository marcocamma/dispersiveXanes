# dispersiveXanes

```python
# INSTALL THE NEEDED PYTHON MODULES
# 1. Go in a folder where you will install the things (cd ~/my_folder)
# 2. git clone https://github.com/marcocamma/dispersiveXanes
# 3. A number of packages (x3py, datastorage, mcutils) written by one of us (m.c.) are included 
# 4. if you want to update go to the right folder (for example cd ~/my_folder/dispersiveXanes) and type git pull.

# BEFORE GETTING STARTED
# 1. if at LCLS load the anaconda session
# source ~marcoc/setups/ana-marco3k-setup.sh
# 2. have a look to the HOW-TO

# GET STARTED
# 1. start ipython (ipython3)
# 2. tell python to use look for modules in the folder
# 3. import sys; sys.path.insert(0,"~/my_folder")

# these are the important files:
# 1. dispersiveXanes_alignment    (deals with images)
# 2. xanes_analyzeRun.py          (deals with run and images reading)
# 3. xppl37_calibration.py        (functions for spectrometer calibration)
# 4. xppl37_theta_scan.py         (functions with theta scans)
# 5. xppl37_spectra.py            (functions for calculation of absorption spectra for IN/OUT scans, etc.)

%matplotlib nbagg
import matplotlib
import matplotlib.pylab as plt
matplotlib.style.use("ggplot")
import pprint
import numpy as np
np.warnings.simplefilter('ignore')

import xanes_analyzeRun
import alignment

#Doing first alignment on "hole"

# define starting parameters for analysis; passed directly to iminuit so things like
# limits, or fix_scalex=True, etc. can be used
pars = dict( scalex = 0.6, intensity = 0.1, iblur1=2,fix_iblur1 = False )
# default parameters can be found in alignment.g_fit_default_kw
# you can have a look by uncommenting the following line:
# pprint.pprint(alignment.g_fit_default_kw)

# define the run object
#### NOTE : for xpp un-focused run: swapx=False,swapy=False
#### NOTE : for xpp    focused run: swapx=False,swapy=True
r = xanes_analyzeRun.AnalyzeRun(190,initAlign=pars,swapx=True,swapy=False)

# data are d.spec1 and d.spec2 (spec1 is the one **upbeam**)
# align one shot

# show = True: show only output; showInit=True: show also starting parameters
r0fit=r.doShot(shot=0,calib=0,showInit=True,doFit=True)

# save as default transformation for run (used when reloading without initAlign keywork)
r.saveTransform();

# do more shots without fitting (using last r.initAlign)
# the return value is a list with lots of stuff for each shot
res = r.doShots(slice(100),doFit=False)
print(list(res.keys()))

print(list(res["parameters"].keys()))

alignment.plotRatios(res["ratio"])
ref = np.nanmedian(res["ratio"],axis=0)
trash = plt.xlim(400,600)
trash = plt.ylim(0,2)


# analyze another run using previous alignment

rShot = xanes_analyzeRun.AnalyzeRun(192,initAlign="mecl3616_init_pars/run0190_transform.npy",swapx=True,swapy=False)

out = rShot.doShots(slice(0,5))
ratios = out["ratio"]
plt.figure()
for i,r in enumerate(ratios):
  plt.plot(r/ref,label="Shot %d"%i)
trash = plt.ylim(0,1)
trash = plt.legend(loc=2)

# save results in hdf file for Andy's happiness
rShot.save(overwrite=True)

```
