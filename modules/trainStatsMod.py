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
import matplotlib.dates as mdates
from datetime import datetime
from obspy import read_inventory, UTCDateTime
from obspy.clients.filesystem.sds import Client

# ============================================================
# Helpers: reading and preprocessing and plotting
# ============================================================
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

# ============================================================
# Helpers: spectrogram and train detector
# ============================================================
def computeSpectrogram(x, fs, win_sec=20.0, overlap_frac=0.5):
    nperseg = int(win_sec * fs)
    noverlap = int(nperseg * overlap_frac)

    f, t, Sxx = sp.spectrogram(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
        mode='psd'
    )

    Sxx[Sxx <= 0] = np.finfo(float).tiny
    return f, t, Sxx


def buildTrainMaskFromTrace(x, fs, params):
    nperseg = int(params["spec_win_sec"] * fs)
    noverlap = int(nperseg * params["spec_overlap_frac"])

    f, t, Sxx = sp.spectrogram(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
        mode='psd'
    )

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


def getTrainIntervals(tCols, maskCols, readStart):
    if len(tCols) < 2:
        return []

    dt = np.median(np.diff(tCols))
    events = []
    inTrain = False
    trainStart = None
    colStart = None

    for k, flag in enumerate(maskCols):
        if flag and not inTrain:
            t0 = max(0.0, tCols[k] - 0.5 * dt)
            trainStart = readStart + t0
            colStart = k
            inTrain = True

        elif (not flag) and inTrain:
            t1 = max(0.0, tCols[k] - 0.5 * dt)
            trainEnd = readStart + t1
            colEnd = k - 1

            events.append({
                "start_utc": trainStart,
                "end_utc": trainEnd,
                "duration_sec": trainEnd - trainStart,
                "col_start": colStart,
                "col_end": colEnd,
                "n_cols": colEnd - colStart + 1,
            })

            inTrain = False
            trainStart = None
            colStart = None

    if inTrain:
        t1 = tCols[-1] + 0.5 * dt
        trainEnd = readStart + t1
        colEnd = len(maskCols) - 1

        events.append({
            "start_utc": trainStart,
            "end_utc": trainEnd,
            "duration_sec": trainEnd - trainStart,
            "col_start": colStart,
            "col_end": colEnd,
            "n_cols": colEnd - colStart + 1,
        })

    return events


# ============================================================
# Helpers: structures
# ============================================================
def initializeHistEntry(f, histEdges):
    return {
        "f": f,
        "hist_edges": histEdges.copy(),
        "hist_centers": 0.5 * (histEdges[:-1] + histEdges[1:]),
        "hist_counts": np.zeros((len(f), len(histEdges) - 1), dtype=np.int64),
        "n_samples": np.zeros(len(f), dtype=np.int64),
        "p10": None,
        "p50": None,
        "p90": None,
    }


def initializeTrainStatsStruct(f, histEdgesPsd, histEdgesAtt, meta=None):
    return {
        "meta": meta if meta is not None else {},
        "events": [],
        "histograms": {
            "surface_train": initializeHistEntry(f, histEdgesPsd),
            "surface_notrain": initializeHistEntry(f, histEdgesPsd),
            "underground_train": initializeHistEntry(f, histEdgesPsd),
            "underground_notrain": initializeHistEntry(f, histEdgesPsd),
            "attenuation_train": initializeHistEntry(f, histEdgesAtt),
            "attenuation_notrain": initializeHistEntry(f, histEdgesAtt),
        }
    }


def makeEmptyEvent(event_id=None, chunk_id=None):
    return {
        "event_id": event_id,
        "chunk_id": chunk_id,

        "start_utc": None,
        "end_utc": None,
        "duration_sec": None,

        "col_start": None,
        "col_end": None,
        "n_cols": None,

        "score_max": None,
        "score_median": None,

        "bh_peak_psd": None,
        "bh_peak_freq_hz": None,
        "bh_peak_time_utc": None,
        "bh_band_energy": None,
        "bh_band_median_psd": None,
        "bh_pgv": None,
        "bh_pgv_time_utc": None,

        "sf_peak_psd": None,
        "sf_peak_freq_hz": None,
        "sf_peak_time_utc": None,
        "sf_band_energy": None,
        "sf_band_median_psd": None,
        "sf_pgv": None,
        "sf_pgv_time_utc": None,
        
        "att_peak_db": None,
        "att_peak_freq_hz": None,
        "att_peak_time_utc": None,
        "att_band_median_db": None,
    }

def updateHistEntryFromValues(entry, vals, log10_input=False):
    """
    vals: 1D vector over frequency
    If log10_input=False, vals are linear and we histogram log10(vals).
    If log10_input=True, vals are already in log10-space or dB-space matching hist_edges.
    """
    vals = np.asarray(vals).copy()

    if log10_input:
        useVals = vals
    else:
        vals[vals <= 0] = np.finfo(float).tiny
        useVals = np.log10(vals)

    valid = np.isfinite(useVals)
    if not np.any(valid):
        return

    edges = entry["hist_edges"]
    binIdx = np.clip(
        np.searchsorted(edges, useVals[valid], side='right') - 1,
        0,
        len(edges) - 2
    )

    freqIdx = np.where(valid)[0]
    entry["hist_counts"][freqIdx, binIdx] += 1
    entry["n_samples"][freqIdx] += 1


def computePercentilesFromHist(entry, probs=(0.1, 0.5, 0.9), log10_to_linear=False):
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
            if log10_to_linear:
                arr[validRows] = 10 ** arr[validRows]
        out[p] = arr

    entry["p10"] = out[0.1]
    entry["p50"] = out[0.5]
    entry["p90"] = out[0.9]

def highpass_trace(x, fs, f_hp=2.0, order=4):
    sos = sp.butter(order, f_hp, btype='highpass', fs=fs, output='sos')
    return sp.sosfiltfilt(sos, x)

def fillEventStats(ev0, event_id, chunk_id, score, f, tSpec, SxxBh, SxxSf, SxxAttDb, xBh, xSurf, fs,
                   coreStart, trainBand):
    """
    ev0 is the interval dict from getTrainIntervals().
    Returns a fully filled event dictionary.
    """
    ev = makeEmptyEvent(event_id=event_id, chunk_id=chunk_id)

    ev["start_utc"] = ev0["start_utc"]
    ev["end_utc"] = ev0["end_utc"]
    ev["duration_sec"] = ev0["duration_sec"]
    ev["col_start"] = ev0["col_start"]
    ev["col_end"] = ev0["col_end"]
    ev["n_cols"] = ev0["n_cols"]

    c0 = ev["col_start"]
    c1 = ev["col_end"] + 1

    scoreEvent = score[c0:c1]
    ev["score_max"] = float(np.max(scoreEvent))
    ev["score_median"] = float(np.median(scoreEvent))

    # Event submatrices
    Sbh = SxxBh[:, c0:c1]
    Ssf = SxxSf[:, c0:c1]
    Satt = SxxAttDb[:, c0:c1]
    df = np.median(np.diff(f))
    dt = np.median(np.diff(tSpec))

    # frequency band for robust train-energy summary
    band = (f >= trainBand[0]) & (f <= trainBand[1])
    
    # ---- underground peak ----
    i_bh = np.nanargmax(Sbh)
    fi_bh, ti_bh = np.unravel_index(i_bh, Sbh.shape)
    ev["bh_peak_psd"] = float(Sbh[fi_bh, ti_bh])
    ev["bh_peak_freq_hz"] = float(f[fi_bh])
    ev["bh_peak_time_utc"] = coreStart + float(tSpec[c0 + ti_bh])
    ev["bh_band_energy"] = float(np.sum(Sbh[band, :]) * df * dt) if np.any(band) else np.nan
    ev["bh_band_median_psd"] = float(np.median(Sbh[band, :])) if np.any(band) else np.nan
    
    # ---- surface peak ----
    i_sf = np.nanargmax(Ssf)
    fi_sf, ti_sf = np.unravel_index(i_sf, Ssf.shape)
    ev["sf_peak_psd"] = float(Ssf[fi_sf, ti_sf])
    ev["sf_peak_freq_hz"] = float(f[fi_sf])
    ev["sf_peak_time_utc"] = coreStart + float(tSpec[c0 + ti_sf])
    ev["sf_band_energy"] = float(np.sum(Ssf[band, :]) * df * dt) if np.any(band) else np.nan
    ev["sf_band_median_psd"] = float(np.median(Ssf[band, :])) if np.any(band) else np.nan
    
    # ---- attenuation peak ----
    i_att = np.nanargmax(Satt)
    fi_att, ti_att = np.unravel_index(i_att, Satt.shape)
    ev["att_peak_db"] = float(Satt[fi_att, ti_att])
    ev["att_peak_freq_hz"] = float(f[fi_att])
    ev["att_peak_time_utc"] = coreStart + float(tSpec[c0 + ti_att])
    ev["att_band_median_db"] = float(np.median(Satt[band, :])) if np.any(band) else np.nan
    
    # adding the part for PGV
    i0 = max(0, int(np.floor((ev["start_utc"] - coreStart) * fs)))
    i1 = min(len(xBh), int(np.ceil((ev["end_utc"] - coreStart) * fs)))
    if i1 > i0:
        segBh = xBh[i0:i1]
        segSf = xSurf[i0:i1]
    
    if len(segBh) > 0:
        idxBh = np.argmax(np.abs(segBh))
        ev["bh_pgv"] = float(np.max(np.abs(segBh)))
        ev["bh_pgv_time_utc"] = coreStart + (i0 + idxBh) / fs

    if len(segSf) > 0:
        idxSf = np.argmax(np.abs(segSf))
        ev["sf_pgv"] = float(np.max(np.abs(segSf)))
        ev["sf_pgv_time_utc"] = coreStart + (i0 + idxSf) / fs
    
    return ev


def updateTrainNoTrainHistograms(trainStats, f, SxxSf, SxxBh, SxxAttDb, maskCols):
    """
    Update histograms column-by-column using train / no-train mask.
    Sxx arrays are linear PSD for surface / underground and dB for attenuation.
    """
    nCols = SxxBh.shape[1]

    for j in range(nCols):
        isTrain = bool(maskCols[j])

        sfVec = SxxSf[:, j]
        bhVec = SxxBh[:, j]
        attVec = SxxAttDb[:, j]

        if isTrain:
            updateHistEntryFromValues(trainStats["histograms"]["surface_train"], sfVec, log10_input=False)
            updateHistEntryFromValues(trainStats["histograms"]["underground_train"], bhVec, log10_input=False)
            updateHistEntryFromValues(trainStats["histograms"]["attenuation_train"], attVec, log10_input=True)
        else:
            updateHistEntryFromValues(trainStats["histograms"]["surface_notrain"], sfVec, log10_input=False)
            updateHistEntryFromValues(trainStats["histograms"]["underground_notrain"], bhVec, log10_input=False)
            updateHistEntryFromValues(trainStats["histograms"]["attenuation_notrain"], attVec, log10_input=True)
            
def plotTrainNoTrainPSD(trainStats, title_prefix=None):
    hist = trainStats["histograms"]
    meta = trainStats["meta"]

    label_fs = 18
    title_fs = 20
    tick_fs = 15
    legend_fs = 14
    line_w = 2.2
    patch_alpha = 0.18

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)

    # ------------------------------------------------
    # subplot 1: during train
    # ------------------------------------------------
    for key, color, label in [
        ("surface_train", "tab:blue", "Surface"),
        ("underground_train", "tab:red", "Depth"),
    ]:
        entry = hist[key]
        f = entry["f"]
        p10 = entry["p10"]
        p50 = entry["p50"]
        p90 = entry["p90"]

        valid = (
            np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90) &
            (f > 0) & (p10 > 0) & (p50 > 0) & (p90 > 0)
        )

        if np.any(valid):
            axs[0].fill_between(f[valid], p10[valid], p90[valid], color=color, alpha=patch_alpha)
            axs[0].plot(f[valid], p50[valid], color=color, linewidth=line_w, label=label)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlim(0.1, 18)
    axs[0].set_ylim(10**-17,10**-8)
    axs[0].set_xlabel('Frequency (Hz)', fontsize=label_fs)
    axs[0].set_ylabel(r'PSD($\mathrm{m}^2/\mathrm{s}^4/\mathrm{Hz}$)', fontsize=label_fs)
    axs[0].set_title(f'(a) {title_prefix or meta.get("channel","")} : During train', fontsize=title_fs)
    axs[0].tick_params(axis='both', labelsize=tick_fs)
    axs[0].grid(True, which='both', alpha=0.3)
    axs[0].legend(fontsize=legend_fs)

    # ------------------------------------------------
    # subplot 2: during no train
    # ------------------------------------------------
    for key, color, label in [
        ("surface_notrain", "tab:blue", "Surface"),
        ("underground_notrain", "tab:red", "Depth"),
    ]:
        entry = hist[key]
        f = entry["f"]
        p10 = entry["p10"]
        p50 = entry["p50"]
        p90 = entry["p90"]

        valid = (
            np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90) &
            (f > 0) & (p10 > 0) & (p50 > 0) & (p90 > 0)
        )

        if np.any(valid):
            axs[1].fill_between(f[valid], p10[valid], p90[valid], color=color, alpha=patch_alpha)
            axs[1].plot(f[valid], p50[valid], color=color, linewidth=line_w, label=label)

    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlim(0.1, 18)
    axs[1].set_ylim(10**-17,10**-8)
    axs[1].set_xlabel('Frequency (Hz)', fontsize=label_fs)
    #axs[1].set_ylabel(r'PSD($\mathrm{m}^2/\mathrm{s}^4/\mathrm{Hz}$)', fontsize=label_fs)
    axs[1].set_title(f'(b) {title_prefix or meta.get("channel","")} : No train', fontsize=title_fs)
    axs[1].tick_params(axis='both', labelsize=tick_fs)
    axs[1].grid(True, which='both', alpha=0.3)
    axs[1].legend(fontsize=legend_fs)

    plt.show()
    
def plotTrainNoTrainAttenuation(trainStats, title_prefix=None):
    hist = trainStats["histograms"]
    meta = trainStats["meta"]

    label_fs = 18
    title_fs = 20
    tick_fs = 15
    legend_fs = 14
    line_w = 2.2
    patch_alpha = 0.18

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)

    # ------------------------------------------------
    # subplot 1: attenuation during train
    # ------------------------------------------------
    entry = hist["attenuation_train"]
    f = entry["f"]
    p10 = entry["p10"]
    p50 = entry["p50"]
    p90 = entry["p90"]

    valid = np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90) & (f > 0)
    if np.any(valid):
        axs[0].fill_between(f[valid], p10[valid], p90[valid], color='tab:blue', alpha=patch_alpha)
        axs[0].plot(f[valid], p50[valid], color='tab:blue', linewidth=line_w)

    axs[0].set_xscale('log')
    axs[0].set_xlim(0.1, 18)
    axs[0].set_xlabel('Frequency (Hz)', fontsize=label_fs)
    axs[0].set_ylabel('Attenuation (dB)', fontsize=label_fs)
    axs[0].set_title(f'(a) {title_prefix or meta.get("channel","")} : Attenuation during train', fontsize=title_fs)
    axs[0].tick_params(axis='both', labelsize=tick_fs)
    axs[0].grid(True, which='both', alpha=0.3)
    #axs[0].legend(fontsize=legend_fs)

    # ------------------------------------------------
    # subplot 2: attenuation during no train
    # ------------------------------------------------
    entry = hist["attenuation_notrain"]
    f = entry["f"]
    p10 = entry["p10"]
    p50 = entry["p50"]
    p90 = entry["p90"]

    valid = np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90) & (f > 0)
    if np.any(valid):
        axs[1].fill_between(f[valid], p10[valid], p90[valid], color='tab:blue', alpha=patch_alpha)
        axs[1].plot(f[valid], p50[valid], color='tab:blue', linewidth=line_w)

    axs[1].set_xscale('log')
    axs[1].set_xlim(0.1, 18)
    axs[1].set_xlabel('Frequency (Hz)', fontsize=label_fs)
    #axs[1].set_ylabel('Attenuation(dB)', fontsize=label_fs)
    axs[1].set_title(f'(b) {title_prefix or meta.get("channel","")} : Attenuation no train', fontsize=title_fs)
    axs[1].tick_params(axis='both', labelsize=tick_fs)
    axs[1].grid(True, which='both', alpha=0.3)
    #axs[1].legend(fontsize=legend_fs)

    plt.show()

def plot_trainstats_event_attributes(
    trainstats_pickle,
    y_keys=("bh_peak_psd", "duration_sec", "score_max"),
    y_labels=None,
    title_prefix=None,
    marker='o',
    linestyle='-',
    ms=5,
    lw=1.5,
    figsize=(14, 10),
    sharex=True,
    sort_by_time=True,
    logy_keys=None,
):

    # -----------------------------
    # style
    # -----------------------------
    label_fs = 18
    title_fs = 20
    tick_fs = 16

    if logy_keys is None:
        logy_keys = set()
    else:
        logy_keys = set(logy_keys)

    with open(trainstats_pickle, 'rb') as f:
        trainStats = pickle.load(f)

    events = trainStats.get("events", [])
    meta = trainStats.get("meta", {})

    if len(events) == 0:
        raise ValueError("No events found in trainStats file.")

    if len(y_keys) != 3:
        raise ValueError("Please provide exactly 3 y_keys for the (3,1) subplot layout.")

    # default labels = keys
    if y_labels is None:
        y_labels = list(y_keys)
    if len(y_labels) != 3:
        raise ValueError("y_labels must have the same length as y_keys.")

    # -----------------------------
    # extract event times
    # -----------------------------
    times = []
    for ev in events:
        t = ev.get("start_utc", None)
        if t is None:
            times.append(None)
        else:
            # ObsPy UTCDateTime -> Python datetime
            try:
                times.append(t.datetime)
            except AttributeError:
                times.append(t)

    # optional sorting by time
    valid_idx = [i for i, t in enumerate(times) if t is not None]
    if sort_by_time:
        valid_idx = sorted(valid_idx, key=lambda i: times[i])

    times_sorted = [times[i] for i in valid_idx]

    # -----------------------------
    # helper to pull one attribute
    # -----------------------------
    def get_attr_array(key):
        vals = []
        for i in valid_idx:
            v = events[i].get(key, np.nan)
            if v is None:
                v = np.nan
            vals.append(v)
        return np.asarray(vals, dtype=float)

    # -----------------------------
    # plotting
    # -----------------------------
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=sharex, constrained_layout=True)

    panel_labels = ['(a)', '(b)', '(c)']
    if(title_prefix is not None):
        base_title = title_prefix + ' ' + meta.get("station_underground","Train stats") + ' ' + meta.get("channel","Train stats") 
    else:
        base_title = 'Surface ' + meta.get("station_underground","Train stats") + ' ' + meta.get("channel","Train stats")
    #base_title = title_prefix if title_prefix is not None else meta.get("channel", "Train stats")

    for j, (key, ylab) in enumerate(zip(y_keys, y_labels)):
        y = get_attr_array(key)
        
        #axs[j].plot(times_sorted, y, marker=marker, linestyle=linestyle, ms=ms, lw=lw)

        #if key in logy_keys:
        #    positive = np.isfinite(y) & (y > 0)
        #    if np.any(positive):
        #        axs[j].set_yscale('log')
        
        finite = np.isfinite(y)
        x_plot = np.array(times_sorted)[finite]
        y_plot = y[finite]

        if key in logy_keys:
            positive = y_plot > 0
            x_plot = x_plot[positive]
            y_plot = y_plot[positive]

        if len(y_plot) > 0:
            axs[j].set_yscale('log')
            ymin = np.nanmin(y_plot) * 0.8
            axs[j].vlines(x_plot, ymin, y_plot, linewidth=lw)
            axs[j].plot(x_plot, y_plot, 'o', ms=4)
        else:
            axs[j].vlines(x_plot, 0, y_plot, linewidth=lw)
            axs[j].plot(x_plot, y_plot, 'o', ms=4)
        
        axs[j].set_ylabel(ylab, fontsize=label_fs)
        axs[j].set_title(f'{panel_labels[j]}{base_title}', fontsize=title_fs)
        axs[j].grid(True, alpha=0.3)
        axs[j].tick_params(axis='both', labelsize=tick_fs)

    axs[-1].set_xlabel('UTC time', fontsize=label_fs)

    # datetime formatting
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)

    # optional overall figure title
    #fig.suptitle(base_title, fontsize=title_fs + 2)

    plt.show()


