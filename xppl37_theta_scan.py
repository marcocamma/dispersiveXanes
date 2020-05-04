import xanes_analyzeRun
import sys


def doRun(run=56):
    r = xanes_analyzeRun.AnalyzeRun(run, initAlign=56)
    ret = r.analyzeScan(nImagesToFit=5)
    r.save(overwrite=True)


run = 56
if len(sys.argv) > 1:
    run = int(sys.argv[1])
doRun(run)
