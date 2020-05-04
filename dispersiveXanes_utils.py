from __future__ import print_function, division
import mcutils as mc
import joblib
import numpy as np

# /--------\
# |        |
# | UTILS  |
# |        |
# \--------/


def rebin1D(a, shape):
    n0 = a.shape[0] // shape
    sh = shape, n0
    return a[: n0 * shape].reshape(sh).mean(1)


def calcFOM(p1, p2, ratio, threshold=0.1):
    idx = p1 > p1.max() * threshold  # & (p2>p2.max()/10)
    ratio = ratio[idx]
    return ratio.std() / np.abs(ratio.mean())


def getCenterOfMass(img, x=None, axis=0, threshold=0.05):
    img = img.copy()
    if img.ndim == 1:
        img = img[np.newaxis, :]
        axis = 1
    img[img < img.max() * threshold] = 0
    if axis == 1:
        img = img.T
    p = img.mean(1)
    if x is None:
        x = np.arange(img.shape[0])
    return np.sum(x * p) / np.sum(p)


def maskLowIntensity(p1, p2, threshold=0.03, squeeze=True):
    if p1.ndim == 1:
        p1 = p1[np.newaxis, :]
        p2 = p2[np.newaxis, :]
    p1 = np.ma.asarray(p1.copy())
    p2 = np.ma.asarray(p2.copy())
    if threshold is not None:
        m1 = np.nanmax(p1, axis=1)
        m2 = np.nanmax(p2, axis=1)
        # find where each spectrum is smaller than threshold*max_for_that_shot; they will be masked out
        idx1 = p1 < (m1[:, np.newaxis] * threshold)
        # idx2 = p2 < (m2[:,np.newaxis]*threshold)
        idx = idx1  # & idx2
        p1.mask = idx
        p2.mask = idx
    if squeeze:
        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
    p1.fill_value = np.nan
    p2.fill_value = np.nan
    return p1, p2


def ratioOfAverage(p1, p2, threshold=0.03):
    """ 
   p1 and p2 are the energy spectrum. if 2D the first index has to be the shot number
   calculate median ratio taking into account only regions where p1 and p2 are > 5% of the max """
    p1, p2 = maskLowIntensity(p1, p2, threshold=threshold, squeeze=False)
    # using masked array because some pixel will have zero shots contributing
    av1 = np.ma.average(p1, axis=0, weights=p1)
    av2 = np.ma.average(p2, axis=0, weights=p2)
    return av2 / av1


def medianRatio(p1, p2, threshold=0.03):
    """ 
   p1 and p2 are the energy spectrum. if 2D the first index has to be the shot number
   calculate median ratio taking into account only regions where p1 and p2 are > 5% of the max """
    p1, p2 = maskLowIntensity(p1, p2, threshold=threshold, squeeze=False)
    ratio = p2 / p1
    return np.ma.average(ratio, axis=0, weights=p1)
