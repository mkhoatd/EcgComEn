import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wfdb import rdrecord, rdann
import os.path as osp
import neurokit2 as nk
import matplotlib

MIT_DIR = 'mit-bih'
sig_name = '100'
sig_index = 0

if __name__ == '__main__':
    record = rdrecord(osp.join(MIT_DIR, sig_name))
    ann = rdann(record_name=osp.join(MIT_DIR, sig_name), extension='atr')
    sig = record.p_signal[:, sig_index]
    _, rpeaks_detect = nk.ecg_peaks(sig, sampling_rate=360)
    rpeaks_detect: np.ndarray
    rpeaks_detect = rpeaks_detect['ECG_R_Peaks'].astype(int)
    rpeaks_real = ann.sample
    # plot = nk.events_plot([rpeaks_detect, rpeaks_real], sig)
    linestyle = '--'
    color_map = matplotlib.colormaps['rainbow']
    colors = color_map(np.linspace(0, 1, num=2))
    sig = pd.DataFrame({"Signal": sig})
    sig: pd.DataFrame
    sig.plot()
    events = [rpeaks_real, rpeaks_detect]
    legends = ['True peak', 'Detected peak']
    for i, event in enumerate(events):
        for j in event:
            plt.axvline(j, color=colors[i], linestyle=linestyle, label=legends[i])
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
    plt.show()
