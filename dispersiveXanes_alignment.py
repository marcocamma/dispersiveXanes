from __future__ import print_function, division
from skimage import transform as tf
from scipy.ndimage import gaussian_filter1d as g1d
import joblib
import collections
import numpy as np
import matplotlib.pyplot as plt
import time

import mcutils as mc
import dispersiveXanes_utils as utils

# /--------\
# |        |
# | UTILS  |
# |        |
# \--------/

# defaultE = np.arange(1024)*0.12+7060
defaultE = (np.arange(1024) - 512) * 0.189 + 7123
defaultE = 7063.5 + 0.1295 * np.arange(1024)
# done on nov 30 2016, based on xppl3716:r77
defaultE = 7064.5 - 6 + 0.1295 * np.arange(1024)
# done on jun 07 2017, based on reference spectrum from ESRF

__i = np.arange(1024)
__x = (__i - 512) / 512

g_fit_default_kw = dict(
    intensity=1.0,
    error_intensity=0.02,
    transx=0,
    error_transx=3,
    limit_transx=(-400, 400),
    transy=0,
    error_transy=3,
    limit_transy=(-50, 50),
    rotation=0.00,
    error_rotation=0.005,
    limit_rotation=(-0.06, 0.06),
    scalex=1,
    limit_scalex=(0.4, 1.2),
    error_scalex=0.05,
    scaley=1,
    error_scaley=0.05,
    limit_scaley=(0.8, 1.2),
    shear=0.00,
    error_shear=0.001,
    limit_shear=(-0.2, 0.2),
    igauss1cen=512,
    error_igauss1cen=2.0,
    fix_igauss1cen=True,
    igauss1sig=4000.0,
    error_igauss1sig=2.0,
    fix_igauss1sig=True,
    igauss2cen=512,
    error_igauss2cen=2.0,
    fix_igauss2cen=True,
    igauss2sig=4000.0,
    error_igauss2sig=2.0,
    fix_igauss2sig=True,
    iblur1=0,
    limit_iblur1=(0, 20),
    error_iblur1=0.02,
    fix_iblur1=True,
    iblur2=0,
    limit_iblur2=(0, 20),
    error_iblur2=0.02,
    fix_iblur2=True,
)


cmap = plt.cm.viridis if hasattr(plt.cm, "viridis") else plt.cm.gray
kw_2dplot = dict(interpolation="none", aspect="auto", cmap=cmap)

fit_ret = collections.namedtuple(
    "fit_ret",
    [
        "init_pars",
        "final_pars",
        "final_transform1",
        "final_transform2",
        "im1",
        "im2",
        "E",
        "p1",
        "p1_sum",
        "p2",
        "p2_sum",
        "fom",
        "ratio",
        "tneeded",
    ],
)


def findRoi(img, height=100, axis=0):
    c = int(utils.getCenterOfMass(img, axis=axis))
    roi = slice(c - height // 2, c + height // 2)
    return roi


def subtractBkg(imgs, nPix=100, bkg_type="line"):
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, :]
    imgs = imgs.astype(np.float)
    if bkg_type == "line":
        bkg = np.median(imgs[:, :nPix, :], axis=1)
        imgs = imgs - bkg[:, np.newaxis, :]
    elif bkg_type == "corner":
        q1 = imgs[:, :nPix, :nPix].mean(-1).mean(-1)
        imgs[:, :512, :512] -= q1[:, np.newaxis, np.newaxis]
        q2 = imgs[:, :nPix, -nPix:].mean(-1).mean(-1)
        imgs[:, :512, -512:] -= q2[:, np.newaxis, np.newaxis]
        q3 = imgs[:, -nPix:, -nPix:].mean(-1).mean(-1)
        imgs[:, -512:, -512:] -= q3[:, np.newaxis, np.newaxis]
        q4 = imgs[:, -nPix:, :nPix].mean(-1).mean(-1)
        imgs[:, -512:, :512] -= q4[:, np.newaxis, np.newaxis]
    elif bkg_type is None:
        if imgs.ndim == 2:
            imgs = imgs[np.newaxis, :].astype(np.float)
    else:
        print("Background subtraction '%s' Not impleted" % bkg_type)
    return imgs


# /--------------------\
# |                    |
# |     PLOTS & CO.    |
# |                    |
# \--------------------/


def plotShot(
    im1,
    im2,
    transf1=None,
    transf2=None,
    fig=None,
    ax=None,
    res=None,
    E=defaultE,
    save=None,
):
    if transf1 is not None:
        im1 = transf1.transformImage(im1)
    if transf2 is not None:
        im2 = transf2.transformImage(im2)
    if fig is None and ax is None:
        fig = plt.subplots(2, 3, figsize=[7, 5], sharex=True)[0]
        ax = fig.axes
    elif fig is not None:
        ax = fig.axes
    if E is None:
        E = np.arange(im1.shape[1])
    n = im1.shape[0]
    ax[0].imshow(im1, extent=(E[0], E[-1], 0, n), **kw_2dplot)
    ax[1].imshow(im2, extent=(E[0], E[-1], 0, n), **kw_2dplot)
    ax[2].imshow(im1 - im2, extent=(E[0], E[-1], 0, n), **kw_2dplot)
    if res is None:
        p1 = np.nansum(im1, axis=0)
        p2 = np.nansum(im2, axis=0)
        pr = p2 / p1
    else:
        p1 = res.p1
        p2 = res.p2
        pr = res.ratio
    ax[3].plot(E, p1, lw=3)
    ax[4].plot(E, p1, lw=1)
    ax[4].plot(E, p2, lw=3)
    idx = p1 > p1.max() / 10.0
    ax[5].plot(E[idx], pr[idx])
    if res is not None:
        ax[5].set_title("FOM: %.2f" % res.fom)
    else:
        ax[5].set_title("FOM: %.2f" % utils.calcFOM(p1, p2, pr))
    if (save is not None) and (save is not False):
        plt.savefig(save, transparent=True, dpi=500)
    return fig


def plotRatios(r, shot="random", fig=None, E=defaultE, save=None):
    if fig is None:
        fig = plt.subplots(2, 1, sharex=True)[0]
    ax = fig.axes
    n = r.shape[0]
    i = ax[0].imshow(r, extent=(E[0], E[-1], 0, n), **kw_2dplot)
    i.set_clim(0, 1.2)
    if shot == "random":
        shot = np.random.random_integers(0, n - 1)
    ax[1].plot(E, r[shot], label="Shot n %d" % shot)
    ax[1].plot(E, np.nanmedian(r[:10], axis=0), label="median 10 shots")
    ax[1].plot(E, np.nanmedian(r, axis=0), label="all shots")
    ax[1].legend()
    ax[1].set_ylim(0, 1.5)
    ax[1].set_xlabel("Energy")
    ax[1].set_ylabel("Transmission")
    ax[0].set_ylabel("Shot num")
    if (save is not None) and (save is not False):
        plt.savefig(save, transparent=True, dpi=500)


def plotSingleShots(
    r, nShots=10, fig=None, E=defaultE, save=None, ErangeForStd=(7090, 7150)
):
    if fig is None:
        fig = plt.subplots(2, 1, sharex=True)[0]
    ax = fig.axes
    for i in range(nShots):
        ax[0].plot(E, r[i] + i)
    ax[0].set_ylim(0, nShots + 0.5)
    av = (1, 3, 10, 30, 100)
    good = np.nanmedian(r, 0)
    for i, a in enumerate(av):
        m = np.nanmedian(r[:a], 0)
        idx = (E > ErangeForStd[0]) & (E < ErangeForStd[1])
        fom = np.nanstd(m[idx] / good[idx])
        print("n shots %d, std %.2f" % (a, fom))
        ax[1].plot(E, m + i, label="%d shots, std :%.2f" % (a, fom))
    ax[1].legend()
    ax[1].set_ylim(0, len(av) + 0.5)
    ax[1].set_xlabel("Energy")
    ax[1].set_ylabel("Transmission")
    if (save is not None) and (save is not False):
        plt.savefig(save, transparent=True, dpi=500)


# /--------------------\
# |                    |
# |  TRANSFORM & CO.   |
# |                    |
# \--------------------/


def transformImage(
    img,
    transform=None,
    iblur=None,
    intensity=1,
    igauss=None,
    orderRotation=1,
    show=False,
):
    """ Transform is the geometrical (affine) transform)
      blur is a tuple (blurx,blury); if a single value used for energy axis
      i    is to correct the itensity
      igauss is to multiply the intensity with a gaussian along the energy axis (=0); if a single assumed centered at center of image, or has to be a tuple (cen,witdth)
  """
    if transform is not None and not np.all(transform.params == np.eye(3)):
        try:
            t = np.linalg.inv(transform.params)
            i = tf._warps_cy._warp_fast(img, t, order=orderRotation)
        except np.linalg.LinAlgError:
            print("Image transformation failed, returning original image")
            i = img.copy()
    else:
        i = img.copy()
    i *= intensity
    if iblur is not None:
        if isinstance(iblur, (int, float)):
            iblur = (iblur, None)
        if (iblur[0] is not None) and (iblur[0] > 0):
            i = g1d(i, iblur[0], axis=1)
        if (iblur[1] is not None) and (iblur[1] > 0):
            i = g1d(i, iblur[1], axis=0)
    if igauss is not None:
        if isinstance(igauss, (int, float)):
            igauss = (__i[-1] / 2, igauss)
            # if one parameter only, assume it is centered on image
        g = mc.gaussian(__i, x0=igauss[0], sig=igauss[1], normalize=False)
        i *= g
    if show:
        plotShot(img, i)
    return i


class SpecrometerTransformation(object):
    def __init__(
        self,
        translation=(0, 0),
        scale=(1, 1),
        rotation=0,
        shear=0,
        intensity=1,
        igauss=None,
        iblur=None,
    ):
        self.affineTransform = getTransform(
            translation=translation, scale=scale, rotation=rotation, shear=shear
        )
        self.intensity = intensity
        self.igauss = igauss
        self.iblur = iblur

    def update(self, **kw):
        # current transformation, necessary because skimage transformation do not support
        # setting of attributes
        names = ["translation", "scale", "rotation", "shear"]
        trans_dict = dict([(n, getattr(self.affineTransform, n)) for n in names])
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k in names:
                trans_dict[k] = v
        self.affineTransform = getTransform(**trans_dict)

    def transformImage(self, img, orderRotation=1, show=False):
        return transformImage(
            img,
            self.affineTransform,
            iblur=self.iblur,
            intensity=self.intensity,
            igauss=self.igauss,
            orderRotation=orderRotation,
            show=show,
        )


def saveAlignment(fname, transform, roi1, roi2, swap):
    np.save(fname, dict(transform=transform, roi1=roi1, roi2=roi2, swap=swap))


def loadAlignment(fname):
    return np.load(fname).item()


def getBestTransform(results):
    # try to see if it is unravelled
    if not hasattr(results, "_fields"):
        results = unravel_results(results)
    # find best based on FOM
    idx = np.nanargmin(np.abs(results.fom))
    parnames = results.init_pars.keys()
    bestPars = dict()
    for n in parnames:
        bestPars[n] = results.final_pars[n][idx]
        if n.find("limit_") == 0:
            if bestPars[n][0] is None:
                bestPars[n] = None
            else:
                bestPars[n] = tuple(bestPars[n])
    return bestPars, np.nanmin(np.abs(results.fom))


def unravel_results(res, nSaveImg="all"):
    final_pars = dict()
    # res[0].fit_result is a list if we are trying to unravel retult from getShots
    if isinstance(res[0].init_pars, list):
        parnames = res[0].init_pars[0].get_keys()
    else:
        parnames = res[0].init_pars.keys()
    final_pars = dict()
    init_pars = dict()
    for n in parnames:
        if n.find("limit_") == 0:
            final_pars[n] = np.vstack([r.final_pars[n] for r in res])
            init_pars[n] = np.vstack([r.init_pars[n] for r in res])
        else:
            final_pars[n] = np.hstack([r.final_pars[n] for r in res])
            init_pars[n] = np.hstack([r.init_pars[n] for r in res])
    im1 = np.asarray([r.im1 for r in res if r.im1.shape[0] != 0])
    im2 = np.asarray([r.im2 for r in res if r.im2.shape[0] != 0])
    if nSaveImg != "all":
        im1 = im1[:nSaveImg].copy()
        # copy is necessary to free the original memory
        im2 = im2[:nSaveImg].copy()
        # of original array
    return fit_ret(
        init_pars=init_pars,
        final_pars=final_pars,
        final_transform1=[r.final_transform1 for r in res],
        final_transform2=[r.final_transform2 for r in res],
        im1=im1,
        im2=im2,
        E=defaultE,
        p1=np.vstack([r.p1 for r in res]),
        p1_sum=np.hstack([r.p1_sum for r in res]),
        p2=np.vstack([r.p2 for r in res]),
        p2_sum=np.hstack([r.p2_sum for r in res]),
        ratio=np.vstack([r.ratio for r in res]),
        fom=np.hstack([r.fom for r in res]),
        tneeded=np.hstack([r.tneeded for r in res]),
    )


def getTransform(translation=(0, 0), scale=(1, 1), rotation=0, shear=0):
    t = tf.AffineTransform(
        scale=scale, rotation=rotation, shear=shear, translation=translation
    )
    return t


def findTransform(p1, p2, ttype="affine"):
    return tf.estimate_transform(ttype, p1, p2)


# __i = np.arange(2**12)/2**11
# def _transformToIntensitylDependence(transform):
#  c = 1.
#  if hasattr(transform,"i_a"):
#    c += transform.i_a*_i
#  return c


# /--------\
# |        |
# |   FIT  |
# |        |
# \--------/


def transformIminuit(
    im1,
    im2,
    init_transform=dict(),
    show=False,
    verbose=True,
    zeroThreshold=0.0,
    doFit=True,
    err = 3,
):
    import iminuit

    assert im1.dtype == im2.dtype
    t0 = time.time()
    # create local copy we can mess up with
    im1_toFit = im1.copy()
    im2_toFit = im2.copy()

    # set anything below the zeroThreshold of the max to zero (one of the two opal is noisy)
    im1_toFit[im1_toFit < im1.max() * zeroThreshold] = 0
    im2_toFit[im2_toFit < im2.max() * zeroThreshold] = 0
    p1 = im1.mean(0)
    p2 = im2.mean(0)

    def transforms(
        intensity,
        igauss1cen,
        igauss1sig,
        iblur1,
        scalex,
        scaley,
        rotation,
        transx,
        transy,
        shear,
        igauss2cen,
        igauss2sig,
        iblur2,
    ):
        t1 = SpecrometerTransformation(
            translation=(transx, transy),
            scale=(scalex, scaley),
            rotation=rotation,
            shear=shear,
            intensity=intensity,
            igauss=(igauss1cen, igauss1sig),
            iblur=iblur1,
        )
        t2 = SpecrometerTransformation(igauss=(igauss2cen, igauss2sig), iblur=iblur2)
        return t1, t2

    def model(
        intensity,
        igauss1cen,
        igauss1sig,
        iblur1,
        scalex,
        scaley,
        rotation,
        transx,
        transy,
        shear,
        igauss2cen,
        igauss2sig,
        iblur2,
    ):

        t1, t2 = transforms(
            intensity,
            igauss1cen,
            igauss1sig,
            iblur1,
            scalex,
            scaley,
            rotation,
            transx,
            transy,
            shear,
            igauss2cen,
            igauss2sig,
            iblur2,
        )
        return t1.transformImage(im1_toFit), t2.transformImage(im2_toFit)

    def chi2(
        intensity,
        igauss1cen,
        igauss1sig,
        iblur1,
        scalex,
        scaley,
        rotation,
        transx,
        transy,
        shear,
        igauss2cen,
        igauss2sig,
        iblur2,
    ):

        i1, i2 = model(
            intensity,
            igauss1cen,
            igauss1sig,
            iblur1,
            scalex,
            scaley,
            rotation,
            transx,
            transy,
            shear,
            igauss2cen,
            igauss2sig,
            iblur2,
        )
        d = (i1 - i2) / err
        return np.sum(d * d)

    # set default initial stepsize and limits
    r = im2.mean(0).sum() / im1.mean(0).sum()
    default_kw = g_fit_default_kw.copy()
    # will be used only if not in initpars
    default_kw["intensity"] = r

    init_kw = dict()
    if isinstance(init_transform, dict):
        init_kw = init_transform.copy()
    elif isinstance(init_transform, iminuit._libiminuit.Minuit):
        init_kw = init_transform.fitarg.copy()
    elif isinstance(init_transform, tf.AffineTransform):
        init_kw["transx"], init_kw["transy"] = init_transform.translation
        init_kw["scalex"], init_kw["scaley"] = init_transform.scale
        init_kw["rotation"] = init_transform.rotation
        init_kw["shear"] = init_transform.shear
    if "intensity" in init_kw and init_kw["intensity"] == "auto":
        r = im2.mean(0).sum() / im1.mean(0).sum()
        init_kw["intensity"] = r

    kw = default_kw.copy()
    kw.update(init_kw)
    kw["fix_shear"] = True

    tofix = ("scalex", "scaley", "rotation", "shear")
    kw_tofix = dict([("fix_%s" % p, True) for p in tofix])
    kw.update(kw_tofix)
    imin = iminuit.Minuit(chi2, errordef=1.0, **kw)
    imin.set_strategy(1)
    init_params = imin.fitarg.copy()

    if show:
        i1, i2 = model(*imin.args)
        plotShot(i1, i2)
        fig = plt.gcf()
        fig.text(0.5, 0.9, "Initial Pars")
        plt.draw()
        input("Enter to start fit")

    if doFit:
        imin.migrad()

    pars = imin.fitarg.copy()
    kw_tofree = dict([("fix_%s" % p, False) for p in tofix])
    pars.update(kw_tofree)
    imin = iminuit.Minuit(chi2, errordef=1.0, **pars)
    imin.set_strategy(1)

    if doFit:
        imin.migrad()

    final_params = imin.fitarg.copy()
    t1, t2 = transforms(*imin.args)
    i1, i2 = model(*imin.args)

    i1 = t1.transformImage(im1)
    i2 = t2.transformImage(im2)
    if show:
        plotShot(i1, i2)
        fig = plt.gcf()
        fig.text(0.5, 0.9, "Final Pars")

    p1 = np.nansum(i1, axis=0)
    p2 = np.nansum(i2, axis=0)
    r = p2 / p1
    idx = p1 > np.nanmax(p1) / 10.0
    fom = utils.calcFOM(p1, p2, r)
    return fit_ret(
        init_pars=init_params,
        final_pars=final_params,
        final_transform1=t1,
        final_transform2=t2,
        im1=i1,
        im2=i2,
        E=defaultE,
        p1=p1,
        p1_sum=p1.sum(),
        p2=p2,
        p2_sum=p2.sum(),
        ratio=r,
        fom=fom,
        tneeded=time.time() - t0,
    )


# /--------\
# |        |
# | MANUAL |
# | ALIGN  |
# |        |
# \--------/
class GuiAlignment(object):
    def __init__(self, im1, im2, autostart=False):
        self.im1 = im1
        self.im2 = im2
        self.f, self.ax = plt.subplots(1, 2)
        self.ax[0].imshow(im1, aspect="auto")
        self.ax[1].imshow(im2, aspect="auto")
        self.transform = None
        if autostart:
            return self.start()
        else:
            print("Zoom first then use the .start method")

    def OnClick(self, event):
        if event.button == 1:
            if self._count % 2 == 0:
                self.im1_p.append((event.xdata, event.ydata))
                print("Added", event.xdata, event.ydata, "to im1")
            else:
                self.im2_p.append((event.xdata, event.ydata))
                print("Added", event.xdata, event.ydata, "to im2")
            self._count += 1
        elif event.button == 2:
            # neglect middle click
            return
        elif event.button == 3:
            self.done = True
            return
        else:
            return

    def start(self):
        self._nP = 0
        self._count = 0
        self.im1_p = []
        self.im2_p = []
        self.done = False
        cid_up = self.f.canvas.mpl_connect("button_press_event", self.OnClick)
        print("Press right button to finish")
        while not self.done:
            if self._count % 2 == 0:
                print("Select point %d for left image" % self._nP)
            else:
                print("Select point %d for right image" % self._nP)
            self._nP = self._count // 2
            plt.waitforbuttonpress()
        self.im1_p = np.asarray(self.im1_p)
        self.im2_p = np.asarray(self.im2_p)
        self.transform = findTransform(self.im1_p, self.im2_p)
        self.transform.intensity = 1.0
        return self.transform

    def show(self):
        if self.transform is None:
            print("Do the alignment first (with .start()")
            return
        # just to save some tipying
        im1 = self.im1
        im2 = self.im2
        im1_new = transformImage(im1, self.transform)
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(im1, aspect="auto")
        ax[1].imshow(im1_new, aspect="auto")
        ax[2].imshow(im2, aspect="auto")

    def save(self, fname):
        if self.transform is None:
            print("Do the alignment first (with .start()")
            return
        else:
            np.save(fname, self.transform)

    def load(self, fname):
        self.transform = np.load(fname).item()


def getAverageTransformation(res):
    if hasattr(res, "final_pars"):
        res = res.final_pars
    out = dict()
    for k in res.keys():
        if k.find("error_") == 0 or k.find("limit_") == 0 or k.find("fix_") == 0:
            if (type(res[k][0]) == np.ndarray) and res[k][0][0] == None:
                out[k] = None
            else:
                out[k] = res[k][0]
        else:
            out[k] = np.nanmedian(res[k])
    return out
    return t


def checkAverageTransformation(out, imgs1):
    t, inten = getAverageTransformation(out)
    res["ratio_av"] = []
    for shot, values in out.items():
        i = transformImage(imgs1[shot], t)
        p1 = np.nansum(i, axis=0)
        r = p1 / values.p2
        res["ratio_av"].append(r)
    res["ratio_av"] = np.asarray(res["ratio_av"])
    return res


g_lastpars = None


def clearCache():
    globals()["g_lastpars"] = None


def doShot(i1, i2, init_pars, doFit=True, show=False):
    # if g_lastpars is not None: init_pars = g_lastpars
    r = transformIminuit(i1, i2, init_pars, show=show, verbose=False, doFit=doFit)
    return r


def doShots(
    imgs1,
    imgs2,
    initpars,
    nJobs=16,
    doFit=False,
    returnBestTransform=False,
    nSaveImg="all",
):
    clearCache()
    N = imgs1.shape[0]
    pool = joblib.Parallel(backend="threading", n_jobs=nJobs)(
        joblib.delayed(doShot)(imgs1[i], imgs2[i], initpars, doFit=doFit)
        for i in range(N)
    )
    res = unravel_results(pool, nSaveImg=nSaveImg)
    if returnBestTransform:
        bestT, fom = getBestTransform(res)
        print("FOM for best alignment %.2f" % fom)
        return res, bestT
    else:
        return res


#  out = collections.OrderedDict( enumerate(pool) )
#  return out

# /--------\
# |        |
# | TEST   |
# | ALIGN  |
# |        |
# \--------/



if __name__ == "__main__":
    pass
