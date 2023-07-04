import matplotlib.pyplot as plt
from scipy import io
from scipy.signal import butter, filtfilt
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import welch

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_eeg():
    start = 17000
    cut_sec = 1000

    pth = 'D:/치매감지/EEG_fNIRs_dataset/Sorted_Dataset/Sorted_alzh_dataset/NORMAL/s001/EEG/sS029R02_RO.mat'
    data = io.loadmat(pth)['data'][start:start+cut_sec,5:13]

    fig, axes = plt.subplots(
        nrows=8, ncols=1, 
        sharex=True, # sharing properties among x axes
        sharey=True, # sharing properties among y axes 
        figsize=(5, 5))

    colors = ['red', 'darkorange', 'darkkhaki', 'darkgreen', 'cyan', 'blue', 'purple', 'brown']
    labels = ['F3', 'FZ', 'F4', 'C3', 'C4', 'P3', 'PZ', 'P4']
    for ch in range(8):
        idx = int('81' + str(ch+1))
        plt.subplot(idx)
        plt.gca().axes.yaxis.set_visible(False)
        plt.plot(range(cut_sec), butter_bandpass_filter(data[:,ch], lowcut=4, highcut=40, fs=500, order=5), color=colors[ch], label=labels[ch])


    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc = 'lower right', bbox_to_anchor=(1,1), bbox_transform=axes[4].transAxes)


    plt.subplots_adjust(
        wspace=0, # the width of the padding between subplots 
        hspace=0)
    plt.xlim(0,cut_sec)
    ax.tick_params(axis='x',direction='in')
    plt.show()

def plot_fnirs():
    start = 1750
    cut_sec = 41
    # start = 0
    # cut_sec = 2000

    pth = 'D:/치매감지/EEG_fNIRs_dataset/Sorted_Dataset/Sorted_alzh_dataset/NORMAL/s001/fNIRs/p_02_oddball1.mat'
    

    colors = ['red', 'green', 'blue']
    labels = ['Hb', 'HbO', 'THb']

    x = list(np.arange(cut_sec)/8)

    for i, label in enumerate(labels):
        data = io.loadmat(pth)[label] # (2299, 6)
        plt.plot(x, butter_bandpass_filter(data[start:start+cut_sec,0], lowcut=0.01, highcut=0.2, fs=8, order=5), color=colors[i], label=labels[i])


    plt.xlim(0,int(cut_sec/8))
    plt.tick_params(axis='both', direction='in')
    plt.xlabel('Time')
    plt.ylabel('Concentration change')
    plt.legend(labels)
    plt.show()

def plot_psd():
    start = 0
    cut_sec = 30000

    n_seg = 256
    n_overlap = 128

    pth = 'D:/치매감지/EEG_fNIRs_dataset/Sorted_Dataset/Sorted_alzh_dataset/NORMAL/s001/EEG/sS029R02_RO.mat'
    data = io.loadmat(pth)['data'][start:start+cut_sec, 23]
    data = butter_bandpass_filter(data, lowcut=4, highcut=47, fs=500, order=5)
    f, psd = welch(data, fs=500, nperseg=n_seg, noverlap=n_overlap)

    plt.plot(np.abs(psd), color='black')


    plt.tick_params(axis='both', direction='in')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V**2/Hz]')
    plt.xlim(0,50)
    plt.ylim(0)
    plt.show()



if __name__ == "__main__":
    
    # plot_eeg()
    # plot_fnirs()
    plot_psd()