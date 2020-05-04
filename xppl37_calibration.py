import matplotlib.pyplot as plt
import numpy as np

import xanes_analyzeRun
import mcutils as mc

colors = "#a6cee3 #1f78b4 #b2df8a #33a02c #fb9a99 #e31a1c #fdbf6f #ff7f00 #cab2d6 #6a3d9a #ffff99 #b15928".split()


def myc(i):
    return colors[i % len(colors)]


def doRun(run=77):
    """ Since p1 is transformed, for the calibration look at p2 """
    r = xanes_analyzeRun.AnalyzeRun(run)
    r.load()
    calibs = list(r.results.keys())
    calibs.sort()
    p2 = [np.nanmedian(r.results[c].p1, axis=0) for c in calibs]
    p2 = np.asarray(p2)
    ref = np.nanmedian(p2, axis=0)
    # normalize p2
    p2 = p2 / ref
    # remove non-flat part
    _x = np.arange(p2.shape[1], dtype=float)
    p2 = [s - mc.poly_approximant(_x, s, order=20)(_x) for s in p2]
    p2 = np.asarray(p2)
    x = np.asarray([_x for _ in p2])
    x[p2 > -0.15] = np.nan
    x[x < 100] = np.nan
    # below pixel 200 is noise
    pos = np.nanmean(x, axis=1)
    fig, ax = plt.subplots(1, 2, sharey=True)
    xshift = 0.2
    pcalib = []
    for icalib, (xi, p2i) in enumerate(zip(x, p2)):
        ppar = dict(color=myc(icalib))
        ax[0].plot(p2i + icalib * xshift, _x, "--", lw=1, alpha=0.5, **ppar)
        ax[0].plot(p2i + icalib * xshift, xi, lw=2, **ppar)
        pcalib.append(np.nanmedian(xi))
        ax[1].plot(r.scanpos[icalib], pcalib[icalib], "o", **ppar)
    polycal = np.polyfit(pcalib, r.scanpos, 1)
    ax[1].plot(np.polyval(polycal, _x), _x)
    print("polynomial calibration", polycal)
