import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import collections
import dispersiveXanes_alignment as alignment
import dispersiveXanes_utils as utils
import xanes_analyzeRun

parser = argparse.ArgumentParser(description="Process argv")

parser.add_argument("--run", type=int, default=82, help="which run to analyze")
parser.add_argument("--force", action="store_true", help="force calculation")
args = parser.parse_args()


profile_ret = collections.namedtuple("profile_ret", ["run", "p1", "p2", "calibs"])


nice_colors = ["#1b9e77", "#d95f02", "#7570b3"]
gradual_colors = [
    "#014636",
    "#016c59",
    "#02818a",
    "#3690c0",
    "#67a9cf",
    "#a6bddb",
    "#d0d1e6",
]  # , '#ece2f0']


def calcSpectraForRun(
    run=82, calibs="all", realign=False, init="auto", alignCalib=0, force=False
):
    """ Calculate spectra (spec1, spec2) for run based on alignment.

      Alignment can be 'forced' with realign=True.
      In such a case 
        init = run for initial alignment
        alignCalib = calibcycle for alignment
  """
    if init == "auto":
        init = run
    if isinstance(run, int):
        r = xanes_analyzeRun.AnalyzeRun(run, initAlign=init)
    else:
        r = run
    if realign:
        r.doShots(slice(20), calib=refCalib, doFit=True)
        r.saveTransform()
    # next line is used to force calculations in case of realignment
    fname = "auto" if not realign else "thisfiledoesnotexists"
    if force:
        fname = "thisfiledoesnotexists"
    if len(r.results) == 0:
        try:
            r.load(fname)
            print("Loading previously saved results")
        except FileNotFoundError:
            r.analyzeScan(calibs=calibs, nImagesToFit=0, nSaveImg=4)
            r.save(overwrite=True)
    # cannot take the output from r.results because it might have been calculated for
    # a bigger range than asked for.
    if isinstance(calibs, int):
        calibsForOut = (calibs,)
    elif isinstance(calibs, slice):
        calibsForOut = list(range(r.nCalib))[calibs]
    elif calibs == "all":
        calibsForOut = list(r.results.keys())
        calibsForOut.sort()
    else:
        calibsForOut = calibs
    # focused data have one single calibcycle ...
    if len(r.results) > 1:
        p1 = [r.results[calib].p1 for calib in calibsForOut]
        p2 = [r.results[calib].p2 for calib in calibsForOut]
    else:
        calib = list(r.results.keys())[0]
        idx = r.results[calib].fom < 0.5
        p1 = [r.results[calib].p1[idx], r.results[calib].p1[~idx]]
        p2 = [r.results[calib].p2[idx], r.results[calib].p2[~idx]]
    return profile_ret(run=r, p1=p1, p2=p2, calibs=calibsForOut)


def calcSpectraForRefAndSample(
    run=82, refCalibs=slice(None, None, 2), forceSpectraCalculation=False
):
    """ Function to calculate the Spectra with and without (ref) the sample.
      It can analyze two kinds of runs:
      * Single run with that alternates IN and OUT (like run 82)
        in this case use something like:
        calcSpectraForRefAndSample(82,refCalibs=slice(None,None,2)
        or
        calcSpectraForRefAndSample(82,refCalibs=(0,2,4,6))
      * Multiple runs (one with reference and one with sample) like run 155 and 156
        in this case use something like:
        calcSpectraForRefAndSample( run=(155,156) )
        where the first is the reference run and the second the one with the sample.
        in this second case the refCalibs does not play a role
      use forceSpectraCalculation = True, to re-read the images
  """
    if isinstance(run, int):
        run = xanes_analyzeRun.AnalyzeRun(run)
        if isinstance(refCalibs, slice):
            refCalibs = list(range(run.nCalib))[refCalibs]
        if isinstance(refCalibs, int):
            refCalibs = [refCalibs]
        sampleCalibs = [c + 1 for c in refCalibs]
        # need a single call (for sample and ref) to save all calibcycles
        data = calcSpectraForRun(
            run, calibs=refCalibs + sampleCalibs, force=forceSpectraCalculation
        )
        # for focused runs
        if len(data.run.results) == 1:
            ref = profile_ret(
                run=data.run, p1=[data.p1[0]], p2=[data.p2[0]], calibs=[0]
            )
            sample = profile_ret(
                run=data.run, p1=[data.p1[1]], p2=[data.p2[1]], calibs=[0]
            )
        else:
            ref = calcSpectraForRun(run, calibs=refCalibs)
            sample = calcSpectraForRun(run, calibs=sampleCalibs)
    elif isinstance(run, (list, tuple)):
        refRun = xanes_analyzeRun.AnalyzeRun(run[0])
        sampleRun = xanes_analyzeRun.AnalyzeRun(run[1], initAlign=run[0])
        refCalibs = [0]
        sampleCalibs = [0]
        ref = calcSpectraForRun(refRun, calibs=refCalibs, force=forceSpectraCalculation)
        sample = calcSpectraForRun(
            sampleRun, calibs=sampleCalibs, force=forceSpectraCalculation
        )
    return ref, sample


def calcRef(r1, r2, calibs=None, threshold=0.05):
    """ r1 and r2 are list of 2d arrays (nShots,nPixels) for each calibcycle """
    if not isinstance(r1, (tuple, list)):
        r1 = (r1,)
    if not isinstance(r2, (tuple, list)):
        r2 = (r2,)
    if calibs is None:
        calibs = list(range(len(r1)))
    if not isinstance(calibs, (tuple, list)):
        calibs = (calibs,)

    out = collections.OrderedDict()
    out["ratioOfAverage"] = dict()
    out["medianOfRatios"] = dict()
    for p1, p2, n in zip(r1, r2, calibs):
        out["ratioOfAverage"][n] = utils.ratioOfAverage(p1, p2, threshold=threshold)
        out["medianOfRatios"][n] = utils.medianRatio(p1, p2, threshold=threshold)
    # add curves with all calib together
    p1 = np.vstack(r1)
    p2 = np.vstack(r2)
    n = ",".join(map(str, calibs))
    ref1 = utils.ratioOfAverage(p1, p2, threshold=threshold)
    ref2 = utils.medianRatio(p1, p2, threshold=threshold)
    out["ratioOfAverage"][n] = utils.ratioOfAverage(p1, p2, threshold=threshold)
    out["medianOfRatios"][n] = utils.medianRatio(p1, p2, threshold=threshold)
    out["ratioOfAverage"]["all"] = out["ratioOfAverage"][n]
    out["medianOfRatios"]["all"] = out["medianOfRatios"][n]
    return out


def showDifferentRefs(run=82, refCalibs=slice(None, None, 2), threshold=0.05):
    """ example plots showing how stable are the different ways of taking spectra """
    prof = calcSpectraForRun(run, calibs=refCalibs)
    refs = calcRef(prof.p1, prof.p2, calibs=prof.calibs)
    kind_of_av = list(refs.keys())
    fig, ax = plt.subplots(len(kind_of_av) + 1, 1, sharex=True, sharey=True)
    E = prof.run.E
    calibs = list(refs[kind_of_av[0]].keys())
    for ikind, kind in enumerate(kind_of_av):
        for calib in calibs:
            if isinstance(calib, int):
                ax[ikind].plot(E, refs[kind][calib], label="calib %s" % calib)
            else:
                if calibs == "all":
                    continue
                ax[ikind].plot(
                    E,
                    refs[kind][calib],
                    label="calib %s" % calib,
                    lw=2,
                    color="k",
                    alpha=0.7,
                )
                ax[-1].plot(
                    E,
                    refs[kind][calib],
                    label="calib all, %s" % kind,
                    lw=1.5,
                    color=nice_colors[ikind],
                    alpha=0.8,
                )
    for ikind, kind in enumerate(kind_of_av):
        ax[ikind].set_title("Run %d, %s" % (run, kind))
    ax[0].set_ylim(0.88, 1.12)
    ax[0].set_ylim(0.88, 1.12)
    ax[-2].legend()
    ax[-1].legend()
    for a in ax:
        a.grid()


def calcAbs(
    ref, sample=None, threshold=0.05, refKind="medianOfRatios", merge_calibs=True
):
    """ example of use
      ratio = calcAbsForRun(82)
      ratio = calcAbsForRun( (155,156) )
  """
    if sample is None:
        sample = ref
    temp = calcRef(ref.p1, ref.p2, threshold=threshold)
    ref = temp[refKind]["all"]
    if merge_calibs:
        p1 = np.vstack(sample.p1)
        p2 = np.vstack(sample.p2)
        p1, p2 = utils.maskLowIntensity(p1, p2, threshold=threshold)
        ratio = p2 / p1
        ratio = ratio / ref
        Abs = -np.log10(ratio)
    else:
        Abs = []
        p1 = []
        p2 = []
        for _p1, _p2 in zip(sample.p1, sample.p2):
            _p1, _p2 = utils.maskLowIntensity(_p1, _p2, threshold=threshold)
            ratio = _p2 / _p1
            ratio = ratio / ref
            _abs = -np.log10(ratio)
            p1.append(_p1)
            p2.append(_p2)
            Abs.append(_abs)
    return p1, p2, Abs


def calcAbsForRun(
    run=82,
    refCalibs=slice(None, None, 2),
    threshold=0.05,
    refKind="medianOfRatios",
    forceSpectraCalculation=False,
    merge_calibs=True,
):
    """ example of use
      ratio = calcAbsForRun(82)
      ratio = calcAbsForRun( (155,156) )
  """
    ref, sample = calcSpectraForRefAndSample(
        run, refCalibs=refCalibs, forceSpectraCalculation=forceSpectraCalculation
    )
    E = ref.run.E
    p1, p2, Abs = calcAbs(
        ref, sample, threshold=threshold, refKind=refKind, merge_calibs=merge_calibs
    )
    return E, p1, p2, Abs


def showSpectra(
    run=82,
    shots=slice(5),
    calibs=0,
    averageEachCalib=False,
    normalization="auto",
    shifty=1,
    xlim=(7060, 7180),
    showAv=True,
):
    """ averageEachCalib: if True, plot only one (averaged) spectrum per calibcycle
      normalization: if "auto", the max of the spectra that will be plotted will be used
  """
    r = xanes_analyzeRun.AnalyzeRun(run=run)
    r.load()
    calibsSaved = list(r.results.keys())
    calibsSaved.sort()
    res = [r.results[c].p2 for c in calibsSaved]
    if isinstance(calibs, slice):
        res = res[calibs]
    if isinstance(calibs, int):
        res = [res[calibs]]
    avCalibs = [np.nanmedian(spectra, axis=0) for spectra in res]
    if averageEachCalib:
        res = avCalibs
        showAv = False
        # it does not make sense to plot it twice !
    fig, ax = plt.subplots(len(res), 1, sharex=True, sharey=True, squeeze=False)
    if normalization == "auto":
        normalization = np.nanmax([temp[shots] for temp in res])
    for (av, spectra, a) in zip(avCalibs, res, ax[:, 0]):
        spectra_norm = spectra[shots] / normalization
        av_norm = av / normalization
        for i, spectrum in enumerate(spectra_norm):
            color = gradual_colors[i % len(gradual_colors)]
            a.axhline(i * shifty, ls="--", color=color)
            if showAv:
                a.fill_between(
                    r.E, i * shifty, av_norm + i * shifty, color="#d95f0e", alpha=0.4
                )
            print(i)
            a.plot(r.E, spectrum + i * shifty, color=color, lw=2)
    ax[0][0].set_xlim(*xlim)
    ax[0][0].set_title("Run %d" % run)
    if not averageEachCalib:
        ax[0][0].set_ylim(0, shifty * (len(spectra_norm)))


color_ss = "#08519c"
color_av = "#238b45"
color_av_all = "#d95f0e"


def showAbs(
    run=82,
    shots=slice(5),
    normalization="auto",
    shifty=1,
    xlim=(7080, 7180),
    showAv=True,
    showAvOverAll=True,
    smoothWidth=0,
    threshold=0.01,
    filterShot=0.1,
):
    """ normalization: if "auto", the max of the spectra that will be plotted will be used
      filterShot = means that it filters out the filterShot*100 percentile
  """
    E = alignment.defaultE
    _, p1, p2, abs = calcAbsForRun(run=run, threshold=threshold)
    p1_sum = p1.sum(-1)
    if filterShot > 0:
        idx = p1_sum > np.percentile(p1_sum, filterShot * 100)
        p1 = p1[idx]
        p2 = p2[idx]
        abs = abs[idx]
    p1_av = np.nanmean(p1, axis=0)
    p2_av = np.nanmean(p2, axis=0)
    # somehow nanmedian screws up when array is too big ... so using nanmean
    abs_av = np.nanmean(abs, axis=0)
    p1 = p1[shots]
    p2 = p2[shots]
    abs = abs[shots]
    if smoothWidth > 0:
        abs = smoothSpectra(E, abs, res=smoothWidth)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False)
    ax = ax[0]
    if normalization == "auto":
        normalization = np.nanmax(p1)
    for ishot, (s1, s2, a) in enumerate(zip(p1, p2, abs)):
        s1_norm = s1 / normalization
        s2_norm = s2 / normalization
        color = gradual_colors[ishot % len(gradual_colors)]
        ax[0].axhline(ishot * shifty, ls="--", color=color)
        ax[1].axhline(ishot * shifty, ls="--", color=color)
        if showAvOverAll:
            if ishot == 0:
                ax[0].fill_between(
                    E,
                    ishot * shifty,
                    p1_av / normalization + ishot * shifty,
                    color=color_av_all,
                    alpha=0.6,
                )
            ax[1].plot(E, abs_av + ishot * shifty, color=color_av_all, lw=2, zorder=20)
        if showAv:
            ax[1].plot(
                E,
                np.nanmedian(abs, 0) + ishot * shifty,
                color=color_av,
                lw=2,
                zorder=10,
            )
        ax[0].plot(E, s1_norm + ishot * shifty, ls="-", color="0.8", lw=2)
        ax[0].plot(E, s2_norm + ishot * shifty, ls="-", color="0.3", lw=2)
        ax[1].plot(E, a + ishot * shifty, color=color_ss, lw=2)
    ax[0].set_xlim(*xlim)
    ax[0].set_title("Run %s" % str(run))
    ax[1].set_ylabel("Sample Absorption")
    ax[0].set_ylabel("Normalized Spectra")
    ax[0].set_ylim(0, shifty * (p1.shape[0]))
    print(
        "STD of (average over shown shots) - (average over all): %.3f"
        % np.nanstd(np.nanmedian(abs, 0) - abs_av)
    )
    ax[1].set_title(
        "STD of (average_{shown}) - (average_{all}): %.3f"
        % np.nanstd(np.nanmedian(abs, 0) - abs_av)
    )


def showAbsWithSweep(run=(155, 156), first=0, period=150, nSpectra=10, **kwards):
    shots = slice(first, first + period, int(period / nSpectra))
    showAbs(run=run, shots=shots, **kwards)


def smoothSpectra(E, abs_spectra, res=0.5, skip_close=5):
    from scipy import integrate

    if isinstance(abs_spectra, np.ma.MaskedArray):
        abs_spectra = abs_spectra.filled()
    if abs_spectra.ndim == 1:
        abs_spectra = abs_spectra[np.newaxis, :]
    out = np.empty_like(abs_spectra)
    for ispectrum, spectrum in enumerate(abs_spectra):
        idx = np.isfinite(spectrum)
        Eclean = E[idx]
        spectrum_clean = spectrum[idx]
        for i in range(len(E)):
            g = (
                1
                / np.sqrt(2 * np.pi)
                / res
                * np.exp(-(Eclean - E[i]) ** 2 / 2 / res ** 2)
            )
            tointegrate = g * spectrum_clean
            out[ispectrum, i] = integrate.simps(tointegrate, x=Eclean)
        out[ispectrum][~idx] = np.nan
        temp = out[ispectrum].copy()
        for i, o in enumerate(temp):
            if np.any(np.isnan(temp[i - skip_close : i + skip_close])):
                out[ispectrum][i] = np.nan
    return out


def doLongCalc():
    # calcSpectraForRefAndSample(82,forceSpectraCalculation=True)

    # scanning requires a lower level call
    r = xanes_analyzeRun.AnalyzeRun(84)
    r.analyzeScan(
        nShotsPerCalib="all",
        calibs="all",
        nSaveImg=2,
        calibsToFit="even",
        nImagesToFit=3,
    )
    r.save(overwrite=True)
    # calcSpectraForRefAndSample(84,forceSpectraCalculation=False)

    calcSpectraForRefAndSample(96, forceSpectraCalculation=True)
    calcSpectraForRefAndSample((155, 156), forceSpectraCalculation=True)
    # r = calcSpectra(run,refCalibs=refCalib,force=force)


if __name__ == "__main__":
    pass
    # main(args.run,force=args.force)
