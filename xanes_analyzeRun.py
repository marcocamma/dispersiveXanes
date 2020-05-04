import os
import sys
import numpy as np

np.warnings.simplefilter("ignore")
import time
import matplotlib.pyplot as plt
import h5py
import collections
import re

from x3py import x3py
import dispersiveXanes_alignment as alignment
import mcutils as mc

cmap = plt.cm.viridis if hasattr(plt.cm, "viridis") else plt.cm.gray
kw_2dplot = dict(interpolation="none", aspect="auto", cmap=cmap)


g_exp = "mecl3616"
g_exp = "xppl3716"
g_bml = g_exp[:3]

x3py.config.updateBeamline(g_bml)
basedir = os.path.dirname(__file__)
g_folder_init = basedir + "/" + g_exp + "_init_pars/"
g_folder_out = basedir + "/" + g_exp + "_output/"
g_folder_data = basedir + "/" + g_exp + "_data/"
# g_folder_data = "/reg/d/psdm/"+g_bml+"/"+ g_exp +"/hdf5/"  # this would be for LCLS

# set defaults based on experiment
if g_bml == "xpp":
    g_roi_height = 200
    g_swapx = False
    g_swapy = False
else:
    g_roi_height = 100
    g_swapx = True
    g_swapy = False

print("Working on experiment", g_exp, "(beamline %s)" % g_bml)
print(" folder data      →", g_folder_data)
print(" folder init_pars →", g_folder_init)
print(" folder outout    →", g_folder_out)

# g_folder = "/reg/d/psdm/xpp/xppl3716/ftc/hdf5/"


def readDataset(fnameOrRun=7, force=False, doBkgSub=False):
    if isinstance(fnameOrRun, str) and (fnameOrRun[-3:] == "npz"):
        d = x3py.toolsVarious.DropObject()
        temp = np.load(fnameOrRun)
        spec1 = temp["spec1"]
        spec2 = temp["spec2"]
        nS = spec1.shape[0]
        d.spec1 = x3py.toolsDetectors.wrapArray("spec1", spec1, time=np.arange(nS))
        d.spec2 = x3py.toolsDetectors.wrapArray("spec2", spec2, time=np.arange(nS))
    else:
        if isinstance(fnameOrRun, int):
            fnameOrRun = g_folder_data + "/" + g_exp + "-r%04d.h5" % fnameOrRun
        d = x3py.Dataset(
            fnameOrRun, detectors=["opal0", "opal1", "fee_spec", "opal2", "ebeam"]
        )
        if g_bml == "xpp":
            d.spec1 = d.opal0
            d.spec2 = d.opal1
        else:
            d.spec1 = d.fee_spec
            d.spec2 = d.opal2
    if not hasattr(d, "scan"):
        d.scan = x3py.toolsVarious.DropObject()
        d.scan.scanmotor0_values = [0]
    return d


def getCenter(img, axis=0, threshold=0.05):
    img = img.copy()
    img[img < img.max() * threshold] = 0
    if axis == 1:
        img = img.T
    p = img.mean(1)
    x = np.arange(img.shape[0])
    return int(np.sum(x * p) / np.sum(p))


def showShots(im1, im2):
    nS = im1.shape[0]
    fig, ax = plt.subplots(2, nS, sharex=True, sharey=True)
    if im1.ndim == 3:
        for a, i1, i2 in zip(ax.T, im1, im2):
            a[0].imshow(i1.T, **kw_2dplot)
            a[1].imshow(i2.T, **kw_2dplot)
    else:
        for a, p1, p2 in zip(ax.T, im1, im2):
            a[0].plot(p1)
            a[1].plot(p2)


def sliceToIndices(shot_slice, nShots):
    return list(range(*shot_slice.indices(nShots)))


class AnalyzeRun(object):
    def __init__(self, run, initAlign="auto", swapx=g_swapx, swapy=g_swapy):
        """ swapx → swap x axis of first spectrometer
        swapy → swap y axis of first spectrometer
        initAlign: could be:
           1. None if you want default transformation parameters
           2. a dict if you want to overwrite certain parameters of the default ones
           3. an integer (to look for xppl3716_init_pars/run????_transform.npy)
           4. a file name (that has been previosly saved with r.saveTransform(fname)
    """
        self.data = readDataset(run)
        self.scanpos = self.data.scan.scanmotor0_values
        self.nCalib = self.data.spec1.nCalib
        self.nShotsPerCalib = self.data.spec1.lens
        if isinstance(run, str):
            run = int(re.search("\d{3,4}", run).group())
        self.run = run
        self.results = collections.OrderedDict()
        self.swap = (swapx, swapy)
        # self.clearCache()

        d = self.data
        self.spec1 = d.spec1
        # spec1 is the one that is moved
        self.spec2 = d.spec2
        self.E = alignment.defaultE

        try:
            self.loadTransform(initAlign)
        except (AttributeError, FileNotFoundError):
            if initAlign is None:
                print("Set to default transform")
                self.initAlign = self.setDefaultTransform()
            else:
                self.initAlign = initAlign

    def getShots(self, shots=0, calib=None, bkgSub="line", roi=g_roi_height):
        if shots == "all":
            if calib != None:
                shots = slice(0, self.nShotsPerCalib[calib])
            else:
                shots = slice(0, self.data.spec1.nShots)
        # read data
        im1 = self.spec1.getShots(shots, calib=calib)
        im2 = self.spec2.getShots(shots, calib=calib)
        # subtractBkg bkg
        im1 = alignment.subtractBkg(im1, bkg_type=bkgSub)
        im2 = alignment.subtractBkg(im2, bkg_type=bkgSub)
        # rebin and swap im1 if necessary
        if im1.shape[-1] != 1024:
            im1 = mc.rebin(im1, (im1.shape[0], im1.shape[1], 1024))
        if self.swap[0]:
            im1 = im1[:, :, ::-1]
        if self.swap[1]:
            im1 = im1[:, ::-1, :]
        if roi is None:
            pass
        elif isinstance(roi, slice):
            im1 = im1[:, roi, :]
            im2 = im2[:, roi, :]
        elif isinstance(roi, int):
            if not hasattr(self, "roi1"):
                self.roi1 = alignment.findRoi(im1[0], roi)
            if not hasattr(self, "roi2"):
                self.roi2 = alignment.findRoi(im2[0], roi)
            im1 = im1[:, self.roi1, :]
            im2 = im2[:, self.roi2, :]
        return im1, im2

    def guiAlign(self, shot=0, save="auto"):
        im1, im2 = self.getShot(shot)
        gui = alignment.GuiAlignment(im1[0], im2[0])
        input("Enter to start")
        gui.start()
        if save == "auto":
            fname = g_folder_init + "/run%04d_gui_align.npy" % self.run
        else:
            fname = save
        self.initAlign = gui.transform
        gui.save(fname)

    def doShot(
        self,
        shot=0,
        calib=None,
        initpars=None,
        im1=None,
        im2=None,
        doFit=True,
        show=False,
        showInit=False,
        save=False,
        savePlot="auto",
    ):
        if initpars is None:
            initpars = self.initAlign
        if (im1 is None) or (im2 is None):
            im1, im2 = self.getShots(shot, calib=calib)
            im1 = im1[0]
            im2 = im2[0]
        r = alignment.doShot(im1, im2, initpars, doFit=doFit, show=showInit)
        im1 = r.im1
        im2 = r.im2
        self.initAlign = r.final_pars
        if show:
            if savePlot == "auto":
                if not os.path.isdir(g_folder_out):
                    os.makedirs(g_folder_out)
                savePlot = g_folder_out + "/run%04d_calib%s_shot%04d_fit.png" % (
                    self.run,
                    calib,
                    shot,
                )
            alignment.plotShot(im1, im2, res=r, save=savePlot)
        if save:
            self.saveTransform()
        return r

    def doShots(
        self,
        shots=slice(0, 50),
        calib=None,
        initpars=None,
        doFit=False,
        returnBestTransform=False,
        nSaveImg=5,
        nInChunks=250,
    ):
        """
        shots   : slice to define shots to read, use 'all' for all shots in calibcycle
        nSaveImg : save saveImg images in memory (self.results), use 'all' for all
                                 useful for decreasing memory footprint
    """
        if initpars is None:
            initpars = self.initAlign
        if shots == "all":
            shots = list(range(self.nShotsPerCalib[calib]))
        if isinstance(shots, slice):
            nmax = (
                self.nShotsPerCalib[calib]
                if calib is not None
                else self.data.spec1.nShots
            )
            shots = sliceToIndices(shots, nmax)
        chunks = x3py.toolsVarious.chunk(shots, nInChunks)
        ret_chunks = []
        for chunk in chunks:
            s1, s2 = self.getShots(chunk, calib=calib)
            ret_chunk = alignment.doShots(
                s1,
                s2,
                initpars=initpars,
                doFit=doFit,
                returnBestTransform=False,
                nSaveImg=nSaveImg,
            )
            ret_chunks.append(ret_chunk)
            if nSaveImg != "all" and len(chunk) < nSaveImg:
                nSaveImg = 0
        ret = alignment.unravel_results(ret_chunks)
        bestT, fom = alignment.getBestTransform(ret)
        if doFit:
            self.initAlign = bestT
        # keep it for later !
        self.results[calib] = ret
        if returnBestTransform:
            return ret, bestT
        else:
            return ret

    def analyzeScan(
        self,
        initpars=None,
        nShotsPerCalib="all",
        calibs="all",
        calibsToFit="all",
        nImagesToFit=0,
        nSaveImg=5,
    ):
        """ nImagesToFit: number of images to Fit per calibcycle, (int or "all")
        calibs to fit could be 'all','even','odd'
    """
        if initpars is None:
            initpars = self.initAlign
        if calibs == "all":
            calibs = list(range(self.nCalib))
        if isinstance(calibs, slice):
            calibs = list(range(self.nCalib))[calibs]
        nC = len(calibs)
        for ic, calib in enumerate(calibs):
            shots = list(range(self.nShotsPerCalib[calib]))
            if nShotsPerCalib != "all":
                shots = shots[:nShotsPerCalib]
            if nImagesToFit == "all":
                nToFit = self.nShotsPerCalib[calib]
            else:
                nToFit = nImagesToFit
            if calibsToFit == "even" and (calib % 2 == 1):
                nToFit = 0
            if calibsToFit == "odd" and (calib % 2 == 0):
                nToFit = 0
            print("Calib %d, tofit %d" % (calib, nToFit))
            ret = None
            if nToFit > 0:
                ret, bestTransf = self.doShots(
                    shots=shots[:nToFit],
                    calib=calib,
                    doFit=True,
                    initpars=initpars,
                    nSaveImg=nSaveImg,
                    returnBestTransform=True,
                )
                initpars = bestTransf
                self.initAlign = bestTransf
            if nToFit < len(shots):
                ret2 = self.doShots(
                    shots[nToFit:],
                    calib=calib,
                    initpars=initpars,
                    doFit=False,
                    nSaveImg=0,
                )
                if ret is None:
                    ret = ret2
                else:
                    ret = alignment.unravel_results((ret, ret2))
            # print("Memory available 3",x3py.toolsOS.memAvailable())
            self.results[calib] = ret
            # print("Memory available 4",x3py.toolsOS.memAvailable())
            print(
                "Calib cycle %d/%d -> %.3f (best FOM: %.2f)"
                % (ic, nC, self.scanpos[ic], np.nanmin(ret.fom))
            )
        return [self.results[c] for c in calibs]

    def save(self, fname="auto", overwrite=False):
        if len(self.results) == 0:
            print("self.results are empty, returning without saving")
            return
        if not os.path.isdir(g_folder_out):
            os.makedirs(g_folder_out)
        if fname == "auto":
            fname = g_folder_out + "/run%04d_analysis.npz" % self.run
        if os.path.exists(fname) and not overwrite:
            print(
                "File %s exists, **NOT** saving, use overwrite=True is you want ..."
                % fname
            )
            return
        h = dict()
        h["roi1"] = (self.roi1.start, self.roi1.stop)
        h["roi2"] = (self.roi2.start, self.roi2.stop)
        if hasattr(self.data.scan, "scanmotor0"):
            h["scanmot0"] = self.data.scan.scanmotor0
        else:
            h["scanmot0"] = "notascan"
        h["scanpos0"] = self.data.scan.scanmotor0_values
        if hasattr(self.data.scan, "scanmotor1"):
            h["scanmot1"] = self.data.scan.scanmotor1
            h["scanpos1"] = self.data.scan.scanmotor1_values
        h["results"] = self.results
        h["E"] = self.E
        np.savez(fname, **h)
        # h["transform"] = self.initAlign

    def load(self, fname="auto"):
        if fname == "auto":
            fname = g_folder_out + "/run%04d_analysis.npz" % self.run
        temp = np.load(fname)
        self.results = temp["results"].item()
        temp.close()

    def _auto_transform_name(self, run=None, calib=None):
        if run is None:
            run = self.run
        fname = g_folder_init + "/run%04d_transform" % run
        if calib is not None:
            fname = fname + "_c%03d" % calib
        return fname + ".npy"

    def saveTransform(self, fname="auto", calib=None, transform=None):
        if transform is None:
            transform = self.initAlign
        if fname == "auto":
            fname = self._auto_transform_name(calib=calib)
        print("Saving roi and transformation parameter to %s" % fname)
        alignment.saveAlignment(fname, self.initAlign, self.roi1, self.roi2, self.swap)

    def loadTransform(self, fname="auto", calib=None):
        if isinstance(fname, dict):
            raise FileNotFoundError
        if fname == "auto":
            fname = self._auto_transform_name(calib=calib)
        if isinstance(fname, int):
            fname = g_folder_init + "/run%04d_transform.npy" % fname
        if not os.path.exists(fname):
            print("Asked to read %s, but it does not exist" % fname)
        temp = np.load(fname).item()
        self.initAlign = temp["transform"]
        self.roi1 = temp["roi1"]
        self.roi2 = temp["roi2"]
        if "swap" in temp:
            self.swap = temp["swap"]
        print("init transform and ROIs from %s" % fname)

    def clearCache(self):
        del self.roi1
        del self.roi2
        alignment.clearCache()
        # nedded for multiprocessing can leave bad parameters in the cache

    def setDefaultTransform(self):
        # dict( scalex=0.65,rotation=0.0,transx=90, iblur1=4.3,fix_iblur1=False )
        t = alignment.g_fit_default_kw
        self.initAlign = t
        return t


def quick_mec(run, ref=236, divideByRef=False, returnRes=False):
    """ useful to analyze the runs around 140 (done with the focusing """
    ref_run = 236
    h = h5py.File("mecl3616_output/run%04d_analysis.h5" % ref, "r")
    ref = np.nanmean(h["calibNone"]["ratio"][...], axis=0)
    r = AnalyzeRun(run, initAlign=ref, swapx=True, swapy=False)
    res = r.doShots(slice(5), doFit=False)
    ret = res["ratio"] / ref if divideByRef else res["ratio"]
    if returnRes:
        return ret, res
    else:
        return ret


def quickAndDirty(run, nShots=300, returnAll=True, doFit=False):
    """ useful to analyze the runs around 140 (done with the focusing """
    r = AnalyzeRun(run, swap=True, initAlign=g_folder_init + "/run0144_transform.npy")
    res = r.doShots(slice(nShots), doFit=doFit)
    o = alignment.unravel_results(res)
    ref = np.nanmedian(o["ratio"][:40], 0)
    sam = np.nanmedian(o["ratio"][50:], 0)
    if returnAll:
        return sam / ref, o["ratio"] / ref
    else:
        return sam / ref
