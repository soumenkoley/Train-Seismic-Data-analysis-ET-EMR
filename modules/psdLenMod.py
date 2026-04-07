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
# Helpers
# ============================================================
def safe_name(s: str) -> str:
    s = s.strip().replace(' ', '_')
    return re.sub(r'[^A-Za-z0-9_\-\.]', '', s)


def fetch_corrected_trace(archive, ch_cfg, inventories, starttime, endtime, preFilt_default):
    try:
        st = archive.get_waveforms(station=ch_cfg["station"],channel=ch_cfg["channel"],network=ch_cfg["network"],
                                   location=ch_cfg["location"],starttime=starttime,endtime=endtime)
        if not st:
            return None, None

        tr = st[0].copy()
        tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
        fs = float(tr.stats.sampling_rate)

        inv = inventories[ch_cfg["inventory_key"]]
        remove_kwargs = dict(inventory=inv, output=ch_cfg.get("output_unit", "VEL"), zero_mean=True, hide_sensitivity_mismatch_warning=True)
        
        if ch_cfg.get("use_prefilt", False):
            remove_kwargs["pre_filt"] = ch_cfg.get("pre_filt", preFilt_default)

        tr.remove_response(**remove_kwargs)
        tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)

        return tr.data.astype(np.float64), fs

    except Exception as e:
        print(f"Failed for {ch_cfg['station']} {ch_cfg['location']} {ch_cfg['channel']} at {starttime}: {e}")
        return None, None


def first_start_in_interval(interval_start, step):
    return math.ceil(interval_start / step) * step


def initialize_hist_entry(f, hist_edges):
    return {"f": f,
            "hist_edges": hist_edges.copy(),
            "hist_centers": 0.5 * (hist_edges[:-1] + hist_edges[1:]),
            "hist_counts": np.zeros((len(f), len(hist_edges) - 1), dtype=np.int64),
            "n_samples": np.zeros(len(f), dtype=np.int64),
            "p10": None,
            "p50": None,
            "p90": None,
           }


def initialize_pair_hist_struct(psd_lengths, fs, hist_edges_psd, hist_edges_att):
    out = {}
    for L in psd_lengths:
        nperseg = int(round(L * fs))
        dummy = np.zeros(nperseg, dtype=float)

        f, _ = sp.welch(dummy, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, nfft=nperseg,
                        detrend='constant')

        out[L] = {"step_sec": L / 2.0,
                  "psd0": initialize_hist_entry(f, hist_edges_psd),
                  "psd1": initialize_hist_entry(f, hist_edges_psd),
                  "att_db": initialize_hist_entry(f, hist_edges_att),
                 }
    return out


def update_hist_entry_linear_psd(entry, f, Pxx, convert_to_acc=True):
    if convert_to_acc:
        Pxx = Pxx * (2 * np.pi * f) ** 2

    Pxx[Pxx <= 0] = np.finfo(float).tiny
    vals = np.log10(Pxx)

    edges = entry["hist_edges"]
    bin_idx = np.clip(np.searchsorted(edges, vals, side='right') - 1, 0, len(edges) - 2)

    freq_idx = np.arange(len(vals))
    entry["hist_counts"][freq_idx, bin_idx] += 1
    entry["n_samples"] += 1


def update_hist_entry_db(entry, vals_db):
    vals = vals_db.copy()
    vals[~np.isfinite(vals)] = np.nan

    edges = entry["hist_edges"]
    valid = np.isfinite(vals)
    if not np.any(valid):
        return

    bin_idx = np.clip(np.searchsorted(edges, vals[valid], side='right') - 1, 0, len(edges) - 2)
    freq_idx = np.where(valid)[0]

    entry["hist_counts"][freq_idx, bin_idx] += 1
    entry["n_samples"][freq_idx] += 1


def compute_percentiles_from_hist(entry, probs=(0.1, 0.5, 0.9), log10_to_linear=False):
    hist_counts = entry["hist_counts"]
    centers = entry["hist_centers"]

    cdf = np.cumsum(hist_counts, axis=1)
    totals = cdf[:, -1].copy()
    valid_rows = totals > 0

    out = {}
    for p in probs:
        arr = np.full(hist_counts.shape[0], np.nan, dtype=float)
        if np.any(valid_rows):
            cdf_norm = np.zeros_like(cdf, dtype=float)
            cdf_norm[valid_rows] = cdf[valid_rows] / totals[valid_rows, None]
            idx = np.argmax(cdf_norm >= p, axis=1)
            arr[valid_rows] = centers[idx[valid_rows]]
            if log10_to_linear:
                arr[valid_rows] = 10 ** arr[valid_rows]
        out[p] = arr

    entry["p10"] = out[0.1]
    entry["p50"] = out[0.5]
    entry["p90"] = out[0.9]


def process_chunk_for_pair_length(data0, data1, fs, read_start_abs, owner_start_abs, owner_end_abs,
                                  L_sec, pair_entry):
    nperseg = int(round(L_sec * fs))
    step_sec = L_sec / 2.0
    t_start = first_start_in_interval(owner_start_abs, step_sec)

    while t_start < owner_end_abs:
        i0 = int(round((t_start - read_start_abs) * fs))
        i1 = i0 + nperseg

        if i0 < 0 or i1 > len(data0) or i1 > len(data1):
            t_start += step_sec
            continue

        seg0 = data0[i0:i1]
        seg1 = data1[i0:i1]

        if len(seg0) != nperseg or len(seg1) != nperseg:
            t_start += step_sec
            continue

        f0, P0 = sp.welch(seg0, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2,
                          nfft=nperseg, detrend='constant')
        f1, P1 = sp.welch(seg1, fs=fs, window='hann',nperseg=nperseg, noverlap=nperseg // 2,
                          nfft=nperseg, detrend='constant')

        # PSD histograms
        update_hist_entry_linear_psd(pair_entry["psd0"], f0, P0, convert_to_acc=True)
        update_hist_entry_linear_psd(pair_entry["psd1"], f1, P1, convert_to_acc=True)

        # attenuation in dB from acceleration PSD ratio
        P0_acc = P0 * (2 * np.pi * f0) ** 2
        P1_acc = P1 * (2 * np.pi * f1) ** 2
        P0_acc[P0_acc <= 0] = np.finfo(float).tiny
        P1_acc[P1_acc <= 0] = np.finfo(float).tiny

        att_db = 10.0 * np.log10(P1_acc / P0_acc)
        update_hist_entry_db(pair_entry["att_db"], att_db)

        t_start += step_sec


def save_pair_results(pair_hist_struct, output_pickle, pair_cfg, metadata_extra=None):
    payload = {
        "pair_config": pair_cfg,
        "metadata": metadata_extra or {},
        "results": pair_hist_struct,
    }
    with open(output_pickle, 'wb') as f:
        pickle.dump(payload, f)

def plot_prct_bands(output_pickle, lengths_to_plot=(3600, 1000, 500), qty_key = 'PSD', title_prefix=None):

    with open(output_pickle, 'rb') as f:
        payload = pickle.load(f)

    results = payload["results"]
    pair_cfg = payload["pair_config"]

    # plotting style
    label_fs = 20
    title_fs = 24
    tick_fs = 16
    legend_fs = 18
    line_w = 2.2
    patch_alpha = 0.3

    # colors for surface / underground
    surf_color = 'tab:blue'
    bh_color = 'tab:red'

    base_title = title_prefix if title_prefix is not None else pair_cfg["label"]
    titleNo = ['a','b','c']
    
    ncols = len(lengths_to_plot)
    fig, ax = plt.subplots(1, ncols, figsize=(6 * ncols, 7), constrained_layout=True, sharey=True)

    if ncols == 1:
        ax = [ax]
    
    for subplotNo, L in enumerate(lengths_to_plot):
        if L not in results:
            ax[subplotNo].set_visible(False)
            continue
        
        if(qty_key=='PSD'):
            # -------------------------
            # surface PSD
            # -------------------------
            entry_surf = results[L]["psd0"]
            f_s = entry_surf["f"]
            p10_s = entry_surf["p10"]
            p50_s = entry_surf["p50"]
            p90_s = entry_surf["p90"]

            valid_s = (
                np.isfinite(p10_s) & np.isfinite(p50_s) & np.isfinite(p90_s) &
                (f_s > 0) & (p10_s > 0) & (p50_s > 0) & (p90_s > 0)
            )

            # -------------------------
            # underground PSD
            # -------------------------
            entry_bh = results[L]["psd1"]
            f_b = entry_bh["f"]
            p10_b = entry_bh["p10"]
            p50_b = entry_bh["p50"]
            p90_b = entry_bh["p90"]

            valid_b = (
                np.isfinite(p10_b) & np.isfinite(p50_b) & np.isfinite(p90_b) &
                (f_b > 0) & (p10_b > 0) & (p50_b > 0) & (p90_b > 0)
            )

            # plot surface
            if np.any(valid_s):
                ax[subplotNo].fill_between(
                    f_s[valid_s], p10_s[valid_s], p90_s[valid_s],
                    color=surf_color, alpha=patch_alpha, label='Surface 10–90%'
                )
                ax[subplotNo].plot(f_s[valid_s], p50_s[valid_s],
                    color=surf_color)
            

            # plot underground
            if np.any(valid_b):
                ax[subplotNo].fill_between(
                    f_b[valid_b], p10_b[valid_b], p90_b[valid_b],
                    color=bh_color, alpha=patch_alpha, label='Depth 10–90%'
                )
                ax[subplotNo].plot(f_b[valid_b], p50_b[valid_b],
                    color=bh_color)

            ax[subplotNo].set_xscale('log')
            ax[subplotNo].set_yscale('log')
            ax[subplotNo].set_xlim(0.1, 15)
            ax[subplotNo].set_ylim(1e-17, 1e-8)

            ax[subplotNo].set_xlabel('Frequency (Hz)', fontsize=label_fs)
            if subplotNo == 0:
                ax[subplotNo].set_ylabel(r'PSD ($\mathrm{m}^2/\mathrm{s}^4/\mathrm{H}z$)', fontsize=label_fs)

            ax[subplotNo].set_title(f'({titleNo[subplotNo]}) {base_title}, {L} s window', fontsize=title_fs)
            ax[subplotNo].grid(True, which='both', alpha=0.3)
            ax[subplotNo].tick_params(axis='both', labelsize=tick_fs)
            ax[subplotNo].legend(fontsize=legend_fs)
        
        elif(qty_key=='Attn'):
            # -------------------------
            # surface-depth attenuation
            # -------------------------
            entry_surf = results[L]["att_db"]
            f_s = entry_surf["f"]
            p10_s = entry_surf["p10"]
            p50_s = entry_surf["p50"]
            p90_s = entry_surf["p90"]

            valid_s = (
                np.isfinite(p10_s) & np.isfinite(p50_s) & np.isfinite(p90_s) &
                (f_s > 0))

            # plot surface
            if np.any(valid_s):
                ax[subplotNo].fill_between(
                    f_s[valid_s], p10_s[valid_s], p90_s[valid_s],
                    color=surf_color, alpha=patch_alpha, label='Depth/Surface 10–90%'
                )
                ax[subplotNo].plot(f_s[valid_s], p50_s[valid_s],
                    color=surf_color)
            
            ax[subplotNo].set_xscale('log')
            ax[subplotNo].set_xlim(0.1, 15)
            ax[subplotNo].set_ylim(-60, 10)

            ax[subplotNo].set_xlabel('Frequency (Hz)', fontsize=label_fs)
            if subplotNo == 0:
                ax[subplotNo].set_ylabel(r'Attenuation (dB)', fontsize=label_fs)

            ax[subplotNo].set_title(f'({titleNo[subplotNo]}) {base_title}, {L} s window', fontsize=title_fs)
            ax[subplotNo].grid(True, which='both', alpha=0.3)
            ax[subplotNo].tick_params(axis='both', labelsize=tick_fs)
            ax[subplotNo].legend(fontsize=legend_fs)
            
    plt.show()


def plot_pair_results(output_pickle, cha01Depth, title_prefix=None):
    with open(output_pickle, 'rb') as f:
        payload = pickle.load(f)

    results = payload["results"]
    pair_cfg = payload["pair_config"]

    colors = {
        3600: 'tab:blue',
        1000: 'tab:red',
        500: 'tab:orange',
        100: 'tab:green',
        10: 'tab:purple',
    }

    # -----------------------------
    # plotting style
    # -----------------------------
    label_fs = 18
    title_fs = 20
    tick_fs = 15
    legend_fs = 14
    line_w = 2.2
    patch_alpha = 0.18

    def _plot_one_quantity(quantity_key, ylabel, ylabelFlag, ax, fig_title, logy=True):

        for L in sorted(results.keys(), reverse=True):
            entry = results[L][quantity_key]
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

            ax.plot(
                f[valid], p50[valid],
                color=c, linewidth=line_w, label=f'{L} s'
            )

        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        ax.set_xlim(0.1, 15)
        if(quantity_key=='att_db'):
            ax.set_ylim(-50,10)
            ax.set_ylabel(ylabel, fontsize=label_fs)
        else:
            ax.set_ylim(10**-17,10**-8)
            if(ylabelFlag):
                ax.set_ylabel(ylabel + r'($\mathrm{m}^2/\mathrm{s}^4/\mathrm{Hz}$)', fontsize=label_fs)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=label_fs)
        ax.set_title(fig_title, fontsize=title_fs)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(title='PSD length', fontsize=legend_fs, title_fontsize=legend_fs)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    base_title = title_prefix if title_prefix is not None else pair_cfg["label"]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, sharey=True)
    # 00 PSD
    _plot_one_quantity(
        quantity_key="psd0",
        ylabel='PSD',
        ylabelFlag=True,
        ax = axs[0],
        fig_title=f'(a) {base_title} on surface',
        logy=True
    )

    # 01 PSD
    _plot_one_quantity(
        quantity_key="psd1",
        ylabel='PSD',
        ylabelFlag = False,
        ax = axs[1],
        fig_title=f'(b) {base_title} at depth ({cha01Depth} m)',
        logy=True
    )
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    # attenuation / spectral ratio in dB
    _plot_one_quantity(
        quantity_key="att_db",
        ylabel='Attenuation (dB)',
        ylabelFlag = True,
        ax = ax1,
        fig_title=f'{base_title} : Attenuation depth/surface',
        logy=False
    )

def process_one_pair(archive, pair_cfg, inventories, preFilt_default, dateUse, nDays, psd_lengths, chunk_owner_sec,
                     response_pad_sec, hist_edges_psd, hist_edges_att, output_dir):
    total_duration_sec = nDays * 86400
    n_chunks = math.ceil(total_duration_sec / chunk_owner_sec)
    max_psd_len = max(psd_lengths)

    pair_hist_struct = None
    fs_global = None
    
    #for chunk_idx in range(1):
    for chunk_idx in range(n_chunks):
        owner_start_abs = chunk_idx * chunk_owner_sec
        owner_end_abs = min((chunk_idx + 1) * chunk_owner_sec, total_duration_sec)

        read_start_abs = max(0.0, owner_start_abs - response_pad_sec)
        read_end_abs = min(total_duration_sec, owner_end_abs + max_psd_len + response_pad_sec)

        read_start = dateUse + read_start_abs
        read_end = dateUse + read_end_abs

        print(
            f"[{pair_cfg['label']}] chunk {chunk_idx+1}/{n_chunks}: "
            f"owner=[{owner_start_abs:.1f},{owner_end_abs:.1f}) "
            f"read=[{read_start_abs:.1f},{read_end_abs:.1f})"
        )

        data0, fs0 = fetch_corrected_trace(archive, pair_cfg["ch0"], inventories, read_start, read_end, preFilt_default)
        data1, fs1 = fetch_corrected_trace(archive, pair_cfg["ch1"], inventories, read_start, read_end, preFilt_default)

        if data0 is None or data1 is None:
            print(f"[{pair_cfg['label']}] skipping chunk {chunk_idx} because one channel is missing")
            continue

        if fs0 != fs1:
            print(f"[{pair_cfg['label']}] sampling-rate mismatch {fs0} vs {fs1}; skipping chunk")
            continue

        if pair_hist_struct is None:
            pair_hist_struct = initialize_pair_hist_struct(psd_lengths, fs0, hist_edges_psd, hist_edges_att)
            fs_global = fs0
        elif fs0 != fs_global:
            print(f"[{pair_cfg['label']}] sampling rate changed; skipping chunk")
            continue

        for L in psd_lengths:
            process_chunk_for_pair_length(data0=data0, data1=data1, fs=fs0, read_start_abs=read_start_abs,
                                          owner_start_abs=owner_start_abs, owner_end_abs=owner_end_abs,L_sec=L,
                                          pair_entry=pair_hist_struct[L])

    if pair_hist_struct is None:
        print(f"[{pair_cfg['label']}] no data processed")
        return None

    for L in psd_lengths:
        compute_percentiles_from_hist(pair_hist_struct[L]["psd0"], probs=(0.1, 0.5, 0.9), log10_to_linear=True)
        compute_percentiles_from_hist(pair_hist_struct[L]["psd1"], probs=(0.1, 0.5, 0.9), log10_to_linear=True)
        compute_percentiles_from_hist(pair_hist_struct[L]["att_db"], probs=(0.1, 0.5, 0.9), log10_to_linear=False)

    outname = safe_name(f"{pair_cfg['label']}_pair_hist.pkl")
    output_pickle = os.path.join(output_dir, outname)

    metadata_extra = {"psd_lengths": psd_lengths, "chunk_owner_sec": chunk_owner_sec,
                      "response_pad_sec": response_pad_sec, "nDays": int(nDays), "start_time": str(dateUse),}
    save_pair_results(pair_hist_struct, output_pickle, pair_cfg, metadata_extra)
    print(f"[{pair_cfg['label']}] saved to {output_pickle}")
    return output_pickle

