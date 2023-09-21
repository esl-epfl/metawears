import numpy as np
import pickle
import os
from scipy.signal import stft, get_window
from scipy.fft import fft


def stft_func(signal, stft_signal):
    nperseg = 256
    noverlap = 64
    sampling_rate = 256
    freq_resolution = 2
    nfft = sampling_rate * freq_resolution
    cutoff_freq = 80
    freqs, times, spec = stft(signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                              boundary=None, padded=False, window='hanning')
    window = get_window('hanning', Nx=nperseg)
    scale = 1.0 / window.sum()**2
    scale = np.sqrt(scale)

    fft_signal = fft(np.multiply(signal[256-64:256*2-64], window * scale), n=nfft)

    amp = (np.log(np.abs(fft_signal) + 1e-10)).astype(np.float32)
    print(amp[:160] - stft_signal)
    print(stft_signal)


def main():
    save_dir = "/home/amirshah/EPFL/EpilepsyTransformer/input/Siena"
    stft_dir = os.path.join(save_dir, 'task-binary_datatype-eval_STFT')
    file_dir = os.path.join(save_dir, 'task-binary_datatype-eval')
    filenames = os.listdir(file_dir)
    filename = filenames[0]
    with open(os.path.join(file_dir, filename), 'rb') as f:
        data_pkl = pickle.load(f)
        signals = np.asarray(data_pkl['signals'])
        print(filename, signals.shape)

    with open(os.path.join(stft_dir, filename), 'rb') as f:
        data_pkl = pickle.load(f)
        stft_signals = np.asarray(data_pkl['STFT'])
        print(filename, stft_signals.shape)

    stft_func(signals[0], stft_signals[0,:, 1])



if __name__ == '__main__':
    main()