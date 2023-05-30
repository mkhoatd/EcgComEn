#%%
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk

sr = 33

signal = np.load('data.npy')

signal = nk.ecg_process(signal, sampling_rate=sr)[0]['ECG_Clean'].to_numpy()
#%%
_, rpeaks = nk.ecg_peaks(signal, sampling_rate=sr)
_, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=sr, method="peak")
# uncomment for visualizing
# plot = nk.events_plot([waves_peak['ECG_Q_Peaks']], signal)
                        
# plt.show()
rpeaks  = rpeaks['ECG_R_Peaks'].astype(int)
# MIN_DIFF = 13
# rpeak_ranges = []
# for peak in rpeaks:
#     rpeak_ranges.append(signal[peak:peak+MIN_DIFF])

# rpeak_ranges = np.stack(rpeak_ranges[:-1], axis=0)
# for i in range(len(rpeak_ranges)) :
#     max_idx_rel = np.argmax(rpeak_ranges[i])
#     rpeaks[i] = rpeaks[i] + max_idx_rel

# #%%
# # uncomment for visualizing
plot = nk.events_plot([rpeaks], signal)
                        
plt.show()


# #%%
labels = ['poi']*len(rpeaks)


# # # Choose from peak to peak or centered
# # # mode = [20, 20]
mode = 24

image_size = 128

# dpi fix
fig = plt.figure(frameon=False)
dpi = fig.dpi

# fig size / image size
figsize = (image_size / dpi, image_size / dpi)
image_size = (image_size, image_size)


def plot(signal, filename):
    plt.figure(figsize=figsize, frameon=False)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0) # use for generation images with no margin
    plt.plot(signal)
    plt.savefig(filename)

    plt.close()

    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.resize(im_gray, image_size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(filename, im_gray)


if __name__ == "__main__":
    
    if not os.path.exists('output'):
        os.makedirs('output')
    
    for i, (label, peak) in enumerate(zip(labels, rpeaks)):
        if isinstance(mode, int):
            left, right = peak - mode // 2, peak + mode // 2
        else:
            raise Exception("Wrong mode in script beginning")

        if np.all([left > 0, right < len(signal)]):
            datadir = 'output'
            filename = osp.join(datadir, "{}.png".format(peak))

            plot(signal[left:right], filename)

# # %%
