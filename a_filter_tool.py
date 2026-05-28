from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import filtfilt, iirnotch, freqz, butter
from scipy.fftpack import fft, fftshift, fftfreq


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_leads(data, low_cut, high_cut, fs, order=5):
    channel_num = data.shape[0]
    out_leads = []
    for i in range(channel_num):
        one_lead = butter_bandpass_filter(data[i, :], low_cut, high_cut, fs, order=order)
        out_leads.append(one_lead)
    return np.array(out_leads)

#bandpass
def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def do_fft(y, fs):
    Y = fftshift(fft(y, 2 ** 12))
    f = fftshift(fftfreq(2 ** 12, 1 / fs))
    return f, Y

def make_signal(fs, f0, T=250e-3):
    # T is total signal time
    t = np.arange(0, T, 1 / fs)
    y = np.sin(2 * np.pi * f0 * t)
    return t, y

def norch_50(in_signal):
    fs = 256.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 20.0  # Quality factor
    # Design notch filter
    w0 = f0 / (fs / 2)
    b, a = iirnotch(w0, Q)
    y_filt = filtfilt(b, a, in_signal)
    #w, h = freqz(b, a)
    #filt_freq = w * fs / (2 * np.pi)
    return y_filt

def get_notch_filterd_leads(leads_all):
    leads_all_filterd = np.zeros_like(leads_all)
    for ch in range(leads_all.shape[0]):
        leads_all_filterd[ch] = norch_50(leads_all[ch])
    return leads_all_filterd

