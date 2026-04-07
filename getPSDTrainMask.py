#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re
import math
import pickle
import datetime
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

from obspy import read_inventory, UTCDateTime
from obspy.clients.filesystem.sds import Client
from modules import trainMaskMod

# ============================================================
# User configuration
# ============================================================
rootP2 = '/dcache/etseis/et/EMR/Data'
archiveP2 = Client(rootP2)
stnName = sys.argv[1]
compName = sys.argv[2]
dateSt = int(sys.argv[3])
year = int(sys.argv[4])
nDays = int(sys.argv[5])

dateStVec = np.array([dateSt])
yearVec = np.array([year])

inventoryFiles = {
    stnName: '/dcache/etseis/et/EMR/Stations/Stations/NL.'+ stnName + '.xml'}

inventories = {sta: read_inventory(path) for sta, path in inventoryFiles.items()}

preFiltDefault = [0.05, 0.1, 16, 18]

# units in seconds
psdLengths = [3600, 1000, 100]
chunkOwnerSec = 7200
responsePadSec = 600

# histogram settings for PSDs in log10(PSD)
logPsdMin = -24
logPsdMax = -8
nHistBinsPsd = 800
histEdgesPsd = np.linspace(logPsdMin, logPsdMax, nHistBinsPsd + 1)

# histogram settings for attenuation in dB
attDbMin = -80
attDbMax = 40
nHistBinsAtt = 800
histEdgesAtt = np.linspace(attDbMin, attDbMax, nHistBinsAtt + 1)

outputDir = '/data/gravwav/koley/PSDHistBoreholes/PSDLenTrainRemove/'
os.makedirs(outputDir, exist_ok=True)

pairConfigs = [
    {
        "label": stnName + '_' + compName,
        "ch0": {
            "station": stnName,
            "network": "NL",
            "location": "00",
            "channel": compName,
            "inventory_key": stnName,
            "use_prefilt": True,
            "pre_filt": preFiltDefault,
            "output_unit": "VEL",
        },
        "ch1": {
            "station": stnName,
            "network": "NL",
            "location": "01",
            "channel": compName,
            "inventory_key": stnName,
            "use_prefilt": True,
            "pre_filt": preFiltDefault,
            "output_unit": "VEL",
        },
        "depth_diff_m": 300.0,
    },
]


# ============================================================
# Train detector parameters
# ============================================================
trainParams = {
    "spec_win_sec": 20.0,
    "spec_overlap_frac": 0.5,
    "f_band": (6.0, 15.0),
    "z_thresh": 2,
    "min_duration_sec": 20.0,
    "pad_sec": 20.0,
    "mask_ramp_sec": 10.0,
}

# ============================================================
# Main
# ============================================================
day0 = int(dateStVec[0])
year0 = int(yearVec[0])
A = datetime.datetime(year0, 1, 1) + datetime.timedelta(days=day0 - 1)
dateUse = UTCDateTime(A.year, A.month, A.day)

savedFiles = []
for pairCfg in pairConfigs:
    outfile = trainMaskMod.processOnePair(archive=archiveP2, pairCfg=pairCfg, inventories=inventories, preFiltDefault=preFiltDefault, dateUse=dateUse,
                               nDays=nDays, psdLengths=psdLengths, chunkOwnerSec=chunkOwnerSec, responsePadSec=responsePadSec,
                               histEdgesPsd=histEdgesPsd, histEdgesAtt=histEdgesAtt, outputDir=outputDir,
                               trainParams=trainParams)
    if outfile is not None:
        savedFiles.append(outfile)

print("\nSaved files:")
for f in savedFiles:
    print(f)

for f in savedFiles:
    trainMaskMod.plotPairResults(f, titlePrefix=stnName + ' ' + compName)
