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
from modules import psdLenMod

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

##
preFilt_default = [0.05, 0.1, 16, 18]

psd_lengths = [3600, 1000, 100]
chunk_owner_sec = 7200
response_pad_sec = 600

# PSD histogram settings (linear PSD stored in log10 space)
logPsdMin = -24
logPsdMax = -8
nHistBins_psd = 800
hist_edges_psd = np.linspace(logPsdMin, logPsdMax, nHistBins_psd + 1)

# attenuation histogram settings in dB
attDbMin = -80
attDbMax = 40
nHistBins_att = 800
hist_edges_att = np.linspace(attDbMin, attDbMax, nHistBins_att + 1)

output_dir = '/data/gravwav/koley/PSDHistBoreholes/PSDLengthAnalysis/'
os.makedirs(output_dir, exist_ok=True)

pair_configs = [
    {
        "label": stnName + '_' + compName,
        "ch0": {
            "station": stnName,
            "network": "NL",
            "location": "00",
            "channel": compName,
            "inventory_key": stnName,
            "use_prefilt": True,
            "pre_filt": preFilt_default,
            "output_unit": "VEL",
        },
        "ch1": {
            "station": stnName,
            "network": "NL",
            "location": "01",
            "channel": compName,
            "inventory_key": stnName,
            "use_prefilt": True,
            "pre_filt": preFilt_default,
            "output_unit": "VEL",
        },
        "depth_diff_m": cha01Depth,
    },
]

# ============================================================
# Main
# ============================================================
day0 = int(dateStVec[0])
year0 = int(yearVec[0])
A = datetime.datetime(year0, 1, 1) + datetime.timedelta(days=day0 - 1)
dateUse = UTCDateTime(A.year, A.month, A.day)

saved_files = []
for pair_cfg in pair_configs:
    outfile = psdLenMod.process_one_pair(archive=archiveP2, pair_cfg=pair_cfg, inventories=inventories, preFilt_default=preFilt_default,
                                         dateUse=dateUse, nDays=nDays, psd_lengths=psd_lengths, chunk_owner_sec=chunk_owner_sec,
                                         response_pad_sec=response_pad_sec, hist_edges_psd=hist_edges_psd, hist_edges_att=hist_edges_att,
                                         output_dir=output_dir)
    if outfile is not None:
        saved_files.append(outfile)

print("\nSaved files:")
for f in saved_files:
    print(f)

for f in saved_files:
    psdLenMod.plot_pair_results(f,cha01Depth,title_prefix=stnName + ' '+ compName)
    
psdLenMod.plot_prct_bands(outfile, lengths_to_plot=(3600, 1000, 100), qty_key= 'PSD', title_prefix=stnName + ' '+ compName)
psdLenMod.plot_prct_bands(outfile, lengths_to_plot=(3600, 1000, 100), qty_key= 'Attn', title_prefix=stnName + ' '+ compName)


# In[ ]:




