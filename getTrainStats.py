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
from modules import trainStatsMod

# ============================================================
# Configuration
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

chunkCoreSec = 7200
edgePadSec = 600

inventory_files = {
    stnName: '/dcache/etseis/et/EMR/Stations/Stations/NL.' + stnName + '.xml',
}
inventories = {sta: read_inventory(path) for sta, path in inventory_files.items()}

# extract the station location and depth
net = inventories[stnName].select(network="NL", station=stnName, location="01", channel=compName)[0]
sta01 = net.stations[0]
cha01 = sta01.channels[0]
cha01Depth = cha01.depth
print("Latitude:", sta01.latitude)
print("Longitude:", sta01.longitude)
print("Elevation (m):", sta01.elevation)

preFiltDefault = [0.05, 0.1, 16, 18]

cfg = {
    "label": stnName + '_' + compName,
    "surface": {
        "station": stnName,
        "network": "NL",
        "location": "00",
        "channel": compName,
        "inventory_key": stnName,
        "use_prefilt": True,
        "pre_filt": preFiltDefault,
        "output_unit": "VEL",
    },
    "underground": {
        "station": stnName,
        "network": "NL",
        "location": "01",
        "channel": compName,
        "inventory_key": stnName,
        "use_prefilt": True,
        "pre_filt": preFiltDefault,
        "output_unit": "VEL",
    },
}


trainParams = {
    "spec_win_sec": 20.0,
    "spec_overlap_frac": 0.5,
    "f_band": (2.0, 8.0),
    "z_thresh": 2.0,
    "min_duration_sec": 20.0,
    "pad_sec": 0.0,
}

# histogram settings
logPsdMin = -24
logPsdMax = -8
nHistBinsPsd = 800
histEdgesPsd = np.linspace(logPsdMin, logPsdMax, nHistBinsPsd + 1)

attDbMin = -80
attDbMax = 40
nHistBinsAtt = 800
histEdgesAtt = np.linspace(attDbMin, attDbMax, nHistBinsAtt + 1)

output_dir = '/data/gravwav/koley/PSDHistBoreholes/' + stnName + 'Train/'
output_file = os.path.join(output_dir, 'train_stats_multi_chunk.pkl')

# ============================================================
# Main run block
# ============================================================

day0 = int(dateStVec[0])
year0 = int(yearVec[0])

A = datetime.datetime(year0, 1, 1) + datetime.timedelta(days=day0 - 1)
analysisStart = UTCDateTime(A.year, A.month, A.day)

totalDurationSec = nDays * 86400
nChunks = int(np.ceil(totalDurationSec / chunkCoreSec))

trainStats = None
eventCounter = 0

for chunkIdx in range(nChunks):
    coreStartAbs = chunkIdx * chunkCoreSec
    coreEndAbs = min((chunkIdx + 1) * chunkCoreSec, totalDurationSec)

    coreStart = analysisStart + coreStartAbs
    coreEnd = analysisStart + coreEndAbs

    readStart = coreStart - edgePadSec
    readEnd = coreEnd + edgePadSec

    print(f'chunk {chunkIdx+1}/{nChunks}: core=[{coreStart},{coreEnd}] read=[{readStart},{readEnd}]')

    # --------------------------------------------
    # Read padded traces
    # --------------------------------------------
    xSurf, fs0 = trainStatsMod.fetchCorrectedTrace(archiveP2, cfg["surface"], inventories, readStart, readEnd, preFiltDefault)
    xBh, fs1 = trainStatsMod.fetchCorrectedTrace(archiveP2, cfg["underground"], inventories, readStart, readEnd, preFiltDefault)

    if xSurf is None or xBh is None:
        print('Skipping chunk because one trace is missing')
        continue

    if fs0 != fs1:
        print('Skipping chunk because sampling rates do not match')
        continue

    fs = fs1

    # --------------------------------------------
    # Trim padded read back to core interval
    # --------------------------------------------
    i0 = int(edgePadSec * fs)
    i1 = i0 + int((coreEndAbs - coreStartAbs) * fs)

    if len(xSurf) < i1 or len(xBh) < i1:
        print('Skipping chunk because padded read is too short')
        continue

    xSurfCore = xSurf[i0:i1]
    xBhCore = xBh[i0:i1]
    
    xSurfCoreHp = trainStatsMod.highpass_trace(xSurfCore, fs, f_hp=2.0, order=4)
    xBhCoreHp = trainStatsMod.highpass_trace(xBhCore, fs, f_hp=2.0, order=4)

    # --------------------------------------------
    # Spectrograms and train detection on core
    # --------------------------------------------
    fBh, tSpec, SxxBh, score, maskCols = trainStatsMod.buildTrainMaskFromTrace(xBhCore, fs, trainParams)
    fSf, tSpecSf, SxxSf = trainStatsMod.computeSpectrogram(
        xSurfCore, fs,
        win_sec=trainParams["spec_win_sec"],
        overlap_frac=trainParams["spec_overlap_frac"]
    )
    
    # avoid zeroing the DC component
    if fBh[0] == 0:
        fBh = fBh[1:]
        fSf = fSf[1:]
        SxxBh = SxxBh[1:, :]
        SxxSf = SxxSf[1:, :]
    
    # convert to accleration
    SxxSf = SxxSf * (2 * np.pi * fBh[:, None])**2
    SxxBh = SxxBh * (2 * np.pi * fBh[:, None])**2

    if not np.allclose(fBh, fSf) or not np.allclose(tSpec, tSpecSf):
        print('Skipping chunk because surface/underground spectrogram grids do not match')
        continue
    
    # PSD ratio in dB
    SxxAttDb = 10.0 * np.log10(SxxBh / SxxSf)

    trainIntervals = trainStatsMod.getTrainIntervals(tSpec, maskCols, coreStart)

    # --------------------------------------------
    # Initialize structure after first valid chunk
    # --------------------------------------------
    if trainStats is None:
        meta = {
            "station_surface": cfg["surface"]["station"],
            "location_surface": cfg["surface"]["location"],
            "station_underground": cfg["underground"]["station"],
            "location_underground": cfg["underground"]["location"],
            "channel": cfg["surface"]["channel"],
            "fs": fs,
            "spectrogram_win_sec": trainParams["spec_win_sec"],
            "spectrogram_overlap_frac": trainParams["spec_overlap_frac"],
            "train_band_hz": trainParams["f_band"],
            "z_thresh": trainParams["z_thresh"],
            "min_duration_sec": trainParams["min_duration_sec"],
            "pad_sec": trainParams["pad_sec"],
            "edge_pad_sec": edgePadSec,
            "chunk_core_sec": chunkCoreSec,
            "analysis_start": str(analysisStart),
            "nDays": int(nDays),
        }

        trainStats = trainStatsMod.initializeTrainStatsStruct(fBh, histEdgesPsd, histEdgesAtt, meta=meta)

    # --------------------------------------------
    # Fill event records
    # --------------------------------------------
    for ev0 in trainIntervals:
        ev = trainStatsMod.fillEventStats(
            ev0=ev0,
            event_id=eventCounter,
            chunk_id=chunkIdx,
            score=score,
            f=fBh,
            tSpec=tSpec,
            SxxBh=SxxBh,
            SxxSf=SxxSf,
            SxxAttDb=SxxAttDb,
            xBh=xBhCoreHp,
            xSurf=xSurfCoreHp,
            fs=fs,
            coreStart=coreStart,
            trainBand=trainParams["f_band"]
        )
        trainStats["events"].append(ev)
        eventCounter += 1

    # --------------------------------------------
    # Update histograms
    # --------------------------------------------
    trainStatsMod.updateTrainNoTrainHistograms(
        trainStats=trainStats,
        f=fBh,
        SxxSf=SxxSf,
        SxxBh=SxxBh,
        SxxAttDb=SxxAttDb,
        maskCols=maskCols
    )

# ============================================================
# Finalize after loop
# ============================================================
if trainStats is None:
    raise RuntimeError("No valid chunks were processed.")

for key in ["surface_train", "surface_notrain", "underground_train", "underground_notrain"]:
    trainStatsMod.computePercentilesFromHist(trainStats["histograms"][key], probs=(0.1, 0.5, 0.9), log10_to_linear=True)

for key in ["attenuation_train", "attenuation_notrain"]:
    trainStatsMod.computePercentilesFromHist(trainStats["histograms"][key], probs=(0.1, 0.5, 0.9), log10_to_linear=False)

print(f'Total detected train events = {len(trainStats["events"])}')

# save the trainStats
with open(output_file, 'wb') as f:
    pickle.dump(trainStats, f)

print(f'Saved trainStats to {output_file}')

trainStatsMod.plotTrainNoTrainPSD(trainStats, title_prefix=stnName + ' '+ compName)
trainStatsMod.plotTrainNoTrainAttenuation(trainStats, title_prefix=stnName + ' '+ compName)