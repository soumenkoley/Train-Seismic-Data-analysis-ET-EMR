#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import math
import pickle
import datetime
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

from obspy import read_inventory, UTCDateTime
from obspy.clients.filesystem.sds import Client


# ============================================================
# Plot style
# ============================================================
LABEL_FS = 18
TITLE_FS = 18
TICK_FS = 15
LEGEND_FS = 14
LINE_W = 2.0
PATCH_ALPHA = 0.18

# ============================================================
# Helpers
# ============================================================
def safeName(s: str) -> str:
    s = s.strip().replace(' ', '_')
    return re.sub(r'[^A-Za-z0-9_\-\.]', '', s)


def fetchCorrectedTrace(archive, chCfg, inventories, startTime, endTime, preFiltDefault):
    
    """
    returns the trace as a numpy array and the sampling frequency
    returns None, None in case no stream is found or strem is empty for the requested station component etc.
    """
    try:
        st = archive.get_waveforms(station=chCfg["station"], channel=chCfg["channel"], network=chCfg["network"],
                                   location=chCfg["location"], starttime=startTime, endtime=endTime)
        if not st:
            return None, None

        tr = st[0].copy()
        tr.trim(starttime=startTime, endtime=endTime, pad=True, fill_value=0)

        fS = float(tr.stats.sampling_rate)

        inv = inventories[chCfg["inventory_key"]]
        removeKwargs = dict(inventory=inv, output=chCfg.get("output_unit", "VEL"), zero_mean=True,
                             hide_sensitivity_mismatch_warning=True)
        if chCfg.get("use_prefilt", False):
            removeKwargs["pre_filt"] = chCfg.get("pre_filt", preFiltDefault)

        tr.remove_response(**removeKwargs)
        tr.trim(starttime=startTime, endtime=endTime, pad=True, fill_value=0)

        return tr.data.astype(np.float64), fS

    except Exception as e:
        print(f"Failed for {chCfg['station']} {chCfg['location']} {chCfg['channel']} at {startTime}: {e}")
        return None, None


def firstStartInInterval(intervalStart, step):
    return math.ceil(intervalStart/step)*step


def initializeHistEntry(f, histEdges):
    return {"f": f, "hist_edges": histEdges.copy(), "hist_centers": 0.5 * (histEdges[:-1] + histEdges[1:]),
            "hist_counts": np.zeros((len(f), len(histEdges) - 1), dtype=np.int64),
            "n_samples": np.zeros(len(f), dtype=np.int64), "p10": None, "p50": None, "p90": None}


def initializePairHistStruct(psdLengths, fs, histEdgesPsd, histEdgesAtt):
    out = {}
    for L in psdLengths:
        nperseg = int(round(L * fs))
        dummy = np.zeros(nperseg, dtype=float)

        f, _ = sp.welch(dummy, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, nfft=nperseg,
                        detrend='constant')

        out[L] = {
            "step_sec": L / 2.0,
            "psd0_raw": initializeHistEntry(f, histEdgesPsd),
            "psd0_masked": initializeHistEntry(f, histEdgesPsd),
            "psd1_raw": initializeHistEntry(f, histEdgesPsd),
            "psd1_masked": initializeHistEntry(f, histEdgesPsd),
            "att_raw": initializeHistEntry(f, histEdgesAtt),
            "att_masked": initializeHistEntry(f, histEdgesAtt),
        }
    return out


def updateHistEntryLinearPsd(entry, Pxx, convertToAcc=True):
    f = entry["f"]
    if convertToAcc:
        Pxx = Pxx * (2 * np.pi * f) ** 2

    Pxx = Pxx.copy()
    Pxx[Pxx <= 0] = np.finfo(float).tiny
    vals = np.log10(Pxx)

    edges = entry["hist_edges"]
    binIdx = np.clip(np.searchsorted(edges, vals, side='right') - 1, 0, len(edges) - 2)

    freqIdx = np.arange(len(vals))
    entry["hist_counts"][freqIdx, binIdx] += 1
    entry["n_samples"] += 1


def updateHistEntryDb(entry, valsDb):
    vals = valsDb.copy()
    valid = np.isfinite(vals)
    if not np.any(valid):
        return

    edges = entry["hist_edges"]
    binIdx = np.clip(np.searchsorted(edges, vals[valid], side='right') - 1, 0, len(edges) - 2)

    freqIdx = np.where(valid)[0]
    entry["hist_counts"][freqIdx, binIdx] += 1
    entry["n_samples"][freqIdx] += 1


def computePercentilesFromHist(entry, probs=(0.1, 0.5, 0.9), log10ToLinear=False):
    histCounts = entry["hist_counts"]
    centers = entry["hist_centers"]

    cdf = np.cumsum(histCounts, axis=1)
    totals = cdf[:, -1].copy()
    validRows = totals > 0

    out = {}
    for p in probs:
        arr = np.full(histCounts.shape[0], np.nan, dtype=float)
        if np.any(validRows):
            cdfNorm = np.zeros_like(cdf, dtype=float)
            cdfNorm[validRows] = cdf[validRows] / totals[validRows, None]
            idx = np.argmax(cdfNorm >= p, axis=1)
            arr[validRows] = centers[idx[validRows]]
            if log10ToLinear:
                arr[validRows] = 10 ** arr[validRows]
        out[p] = arr

    entry["p10"] = out[0.1]
    entry["p50"] = out[0.5]
    entry["p90"] = out[0.9]


def buildTrainMaskFromTrace(x, fs, params):
    """
    x: time domain trace
    fs = sampling frequency
    params: train detection parameters
    """
    nperseg = int(params["spec_win_sec"] * fs)
    noverlap = int(nperseg * params["spec_overlap_frac"])
    
    # compute the spectrogram using the small windows
    f, t, Sxx = sp.spectrogram(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, detrend='constant',
                               scaling='density', mode='psd')

    Sxx[Sxx <= 0] = np.finfo(float).tiny
    logS = np.log10(Sxx)

    rowMed = np.median(logS, axis=1, keepdims=True)
    rowMad = np.median(np.abs(logS - rowMed), axis=1, keepdims=True)
    rowMad[rowMad <= 0] = 1.0

    Z = (logS - rowMed) / rowMad

    fBand = params["f_band"]
    band = (f >= fBand[0]) & (f <= fBand[1])
    if not np.any(band):
        raise ValueError("Train detector frequency band does not overlap spectrogram frequencies.")

    score = np.median(Z[band, :], axis=0)
    maskCols = score > params["z_thresh"]

    dt = np.median(np.diff(t))
    minCols = max(1, int(np.round(params["min_duration_sec"] / dt)))
    padCols = max(0, int(np.round(params["pad_sec"] / dt)))

    maskClean = np.zeros_like(maskCols, dtype=bool)
    idx = np.where(maskCols)[0]
    if len(idx) > 0:
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for g in groups:
            if len(g) >= minCols:
                i0 = max(0, g[0] - padCols)
                i1 = min(len(maskCols), g[-1] + padCols + 1)
                maskClean[i0:i1] = True

    return f, t, Sxx, score, maskClean


def columnMaskToSampleMask(maskCols, tCols, nSamples, fs):
    """
    Convert spectrogram-column mask to sample-level binary mask.
    """
    sampleMask = np.zeros(nSamples, dtype=float)
    if len(tCols) < 2:
        return sampleMask

    dt = np.median(np.diff(tCols))
    for k, flag in enumerate(maskCols):
        if not flag:
            continue
        t0 = max(0.0, tCols[k] - 0.5 * dt)
        t1 = tCols[k] + 0.5 * dt
        i0 = max(0, int(np.floor(t0 * fs)))
        i1 = min(nSamples, int(np.ceil(t1 * fs)))
        sampleMask[i0:i1] = 1.0

    return sampleMask



def smoothBinaryMask(binaryMask, fs, rampSec):
    """
    Turn a hard 0/1 bad-sample mask into a smooth keep-mask in [0,1].
    bad=1 means remove, keep=1 means preserve.
    """
    keepMask = 1.0 - binaryMask.astype(float)

    rampN = int(round(rampSec * fs))
    if rampN <= 1:
        return keepMask

    kernel = np.hanning(2 * rampN + 1)
    kernel /= kernel.sum()

    badSmooth = np.convolve(binaryMask.astype(float), kernel, mode='same')
    badSmooth = np.clip(badSmooth, 0.0, 1.0)

    keepSmooth = 1.0 - badSmooth
    keepSmooth = np.clip(keepSmooth, 0.0, 1.0)
    return keepSmooth


def maskedWelch(seg, fs):
    nperseg = len(seg)
    f, Pxx = sp.welch(seg, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, nfft=nperseg,
                      detrend='constant')
    return f, Pxx


def maskedWelchWithRenorm(seg, keepMask, fs):
    """
    Compute PSD of masked segment and renormalize by keep-mask RMS energy.
    """
    segMasked = seg * keepMask
    f, Pxx = maskedWelch(segMasked, fs)

    # Renormalization based on mean squared keep-mask.
    # Prevents artificial drop just due to zeroed samples.
    denom = np.mean(keepMask ** 2)
    if denom > 1e-6:
        Pxx = Pxx / denom
    else:
        Pxx[:] = np.nan

    return f, Pxx

def processChunkForPairLength(data0, data1, keepMask, fs, readStartAbs, ownerStartAbs, ownerEndAbs,
                                  LSec, pairEntry):
    nperseg = int(round(LSec * fs))
    stepSec = LSec / 2.0
    tStart = firstStartInInterval(ownerStartAbs, stepSec)
    #print('tStart for L = ' + str(LSec) + ' = ' + str(tStart))
    
    while tStart < ownerEndAbs:
        i0 = int(round((tStart - readStartAbs) * fs))
        i1 = i0 + nperseg

        if i0 < 0 or i1 > len(data0) or i1 > len(data1) or i1 > len(keepMask):
            tStart += stepSec
            continue

        seg0 = data0[i0:i1]
        seg1 = data1[i0:i1]
        segm = keepMask[i0:i1]

        if len(seg0) != nperseg or len(seg1) != nperseg or len(segm) != nperseg:
            tStart += stepSec
            continue

        # raw
        f0, P0 = maskedWelch(seg0, fs)
        f1, P1 = maskedWelch(seg1, fs)

        updateHistEntryLinearPsd(pairEntry["psd0_raw"], P0, convertToAcc=True)
        updateHistEntryLinearPsd(pairEntry["psd1_raw"], P1, convertToAcc=True)

        P0Acc = P0 * (2 * np.pi * f0) ** 2
        P1Acc = P1 * (2 * np.pi * f1) ** 2
        P0Acc[P0Acc <= 0] = np.finfo(float).tiny
        P1Acc[P1Acc <= 0] = np.finfo(float).tiny
        attRawDb = 10.0 * np.log10(P1Acc / P0Acc)
        updateHistEntryDb(pairEntry["att_raw"], attRawDb)

        # masked
        fm0, P0m = maskedWelchWithRenorm(seg0, segm, fs)
        fm1, P1m = maskedWelchWithRenorm(seg1, segm, fs)

        if np.all(np.isfinite(P0m)) and np.all(np.isfinite(P1m)):
            updateHistEntryLinearPsd(pairEntry["psd0_masked"], P0m, convertToAcc=True)
            updateHistEntryLinearPsd(pairEntry["psd1_masked"], P1m, convertToAcc=True)

            P0mAcc = P0m * (2 * np.pi * fm0) ** 2
            P1mAcc = P1m * (2 * np.pi * fm1) ** 2
            P0mAcc[P0mAcc <= 0] = np.finfo(float).tiny
            P1mAcc[P1mAcc <= 0] = np.finfo(float).tiny
            attMaskedDb = 10.0 * np.log10(P1mAcc / P0mAcc)
            updateHistEntryDb(pairEntry["att_masked"], attMaskedDb)

        tStart += stepSec


def savePairResults(pairHistStruct, outputPickle, pairCfg, metadataExtra=None):
    payload = {
        "pair_config": pairCfg,
        "metadata": metadataExtra or {},
        "results": pairHistStruct,
    }
    with open(outputPickle, 'wb') as f:
        pickle.dump(payload, f)


def plotQuantityPanel(ax, results, quantityKey, colors, ylabel, ylabelFlag, titleText, logy=True):
    for L in sorted(results.keys(), reverse=True):
        entry = results[L][quantityKey]
        f = entry["f"]
        p10 = entry["p10"]
        p50 = entry["p50"]
        p90 = entry["p90"]

        valid = np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90) & (f > 0)
        if logy:
            valid = valid & (p10 > 0) & (p50 > 0) & (p90 > 0)

        if not np.any(valid):
            continue

        c = colors.get(L, None)
        #ax.fill_between(f[valid], p10[valid], p90[valid], color=c, alpha=PATCH_ALPHA)
        ax.plot(f[valid], p50[valid], color=c, linewidth=LINE_W, label=f'{L} s')

    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlim(0.1, 16)
    if(quantityKey == 'psd0_raw' or quantityKey == 'psd0_masked' or quantityKey == 'psd1_raw' or quantityKey =='psd1_masked'):
        ax.set_ylim(10**-17, 10**-10)
    else:
        ax.set_ylim(-60,10)
    ax.set_xlabel('Frequency (Hz)', fontsize=LABEL_FS)
    if(ylabelFlag):
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_title(titleText, fontsize=TITLE_FS)
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, title='PSD length', title_fontsize=LEGEND_FS)


def plotPairResults(outputPickle, titlePrefix=None):
    with open(outputPickle, 'rb') as f:
        payload = pickle.load(f)

    results = payload["results"]
    pairCfg = payload["pair_config"]

    colors = {
        3600: 'tab:blue',
        1000: 'tab:red',
        500: 'tab:orange',
        100: 'tab:green',
        10: 'tab:purple',
    }

    baseTitle = titlePrefix if titlePrefix is not None else pairCfg["label"]
    
    # Figure 2: loc 01
    fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)
    plotQuantityPanel(axs2[0], results, "psd1_raw", colors, r'PSD($\mathrm{m}^2/\mathrm{s}^4/\mathrm{Hz}$)', True, f'(a) {baseTitle} raw (depth)', logy=True)
    
    plotQuantityPanel(axs2[1], results, "psd1_masked", colors, r'PSD($\mathrm{m}^2/\mathrm{s}^4/\mathrm{Hz}$)',False, f'(b) {baseTitle} train-masked (depth)', logy=True)
    plt.show()

def processOnePair(archive, pairCfg, inventories, preFiltDefault, dateUse, nDays, psdLengths, chunkOwnerSec,
                     responsePadSec, histEdgesPsd, histEdgesAtt, outputDir, trainParams):
    
    totalDurationSec = nDays * 86400
    nChunks = math.ceil(totalDurationSec / chunkOwnerSec)
    maxPsdLen = max(psdLengths)

    pairHistStruct = None
    fsGlobal = None
    
    #for chunkIdx in range(12):
    for chunkIdx in range(nChunks):
        ownerStartAbs = chunkIdx * chunkOwnerSec
        ownerEndAbs = min((chunkIdx + 1) * chunkOwnerSec, totalDurationSec)
        
        ownerStartUTC = dateUse + ownerStartAbs
        ownerEndUTC = dateUse + ownerEndAbs
        
        readStartAbs = max(0.0, ownerStartAbs - responsePadSec)
        readEndAbs = min(totalDurationSec, ownerEndAbs + maxPsdLen + responsePadSec)

        readStart = dateUse + readStartAbs
        readEnd = dateUse + readEndAbs

        print(f"[{pairCfg['label']}] chunk {chunkIdx+1}/{nChunks}: "
              f"owner=[{ownerStartAbs:.1f},{ownerEndAbs:.1f}) "
              f"read=[{readStartAbs:.1f},{readEndAbs:.1f})")
        
        print('read start UTC ' + str(readStart))
        data0, fs0 = fetchCorrectedTrace(archive, pairCfg["ch0"], inventories, readStart, readEnd, preFiltDefault)
        data1, fs1 = fetchCorrectedTrace(archive, pairCfg["ch1"], inventories, readStart, readEnd, preFiltDefault)

        if data0 is None or data1 is None:
            print(f"[{pairCfg['label']}] skipping chunk {chunkIdx} because one channel is missing")
            continue

        if fs0 != fs1:
            print(f"[{pairCfg['label']}] sampling-rate mismatch {fs0} vs {fs1}; skipping chunk")
            continue

        if pairHistStruct is None:
            pairHistStruct = initializePairHistStruct(psdLengths, fs0, histEdgesPsd, histEdgesAtt)
            fsGlobal = fs0
        elif fs0 != fsGlobal:
            print(f"[{pairCfg['label']}] sampling rate changed; skipping chunk")
            continue

        # train detection from underground channel only
        f, tSpec, Sxx, _, maskCols = buildTrainMaskFromTrace(data1, fs1, trainParams)
        badSampleMask = columnMaskToSampleMask(maskCols, tSpec, len(data1), fs1)
        keepMask = smoothBinaryMask(badSampleMask, fs1, trainParams["mask_ramp_sec"])
        
        for L in psdLengths:
            processChunkForPairLength(data0=data0, data1=data1, keepMask=keepMask, fs=fs0,
                                          readStartAbs=readStartAbs, ownerStartAbs=ownerStartAbs,
                                          ownerEndAbs=ownerEndAbs, LSec=L, pairEntry=pairHistStruct[L])

    if pairHistStruct is None:
        print(f"[{pairCfg['label']}] no data processed")
        return None

    for L in psdLengths:
        computePercentilesFromHist(pairHistStruct[L]["psd0_raw"], probs=(0.1, 0.5, 0.9), log10ToLinear=True)
        computePercentilesFromHist(pairHistStruct[L]["psd0_masked"], probs=(0.1, 0.5, 0.9), log10ToLinear=True)
        computePercentilesFromHist(pairHistStruct[L]["psd1_raw"], probs=(0.1, 0.5, 0.9), log10ToLinear=True)
        computePercentilesFromHist(pairHistStruct[L]["psd1_masked"], probs=(0.1, 0.5, 0.9), log10ToLinear=True)
        computePercentilesFromHist(pairHistStruct[L]["att_raw"], probs=(0.1, 0.5, 0.9), log10ToLinear=False)
        computePercentilesFromHist(pairHistStruct[L]["att_masked"], probs=(0.1, 0.5, 0.9), log10ToLinear=False)

    outname = safeName(f"{pairCfg['label']}_pair_hist_trainmasked.pkl")
    outputPickle = os.path.join(outputDir, outname)

    metadataExtra = {"psd_lengths": psdLengths,"chunk_owner_sec": chunkOwnerSec,"response_pad_sec": responsePadSec,
                      "nDays": int(nDays),"start_time": str(dateUse),"train_params": trainParams}
    
    savePairResults(pairHistStruct, outputPickle, pairCfg, metadataExtra)
    print(f"[{pairCfg['label']}] saved to {outputPickle}")
    return outputPickle

