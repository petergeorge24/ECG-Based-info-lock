import wfdb
from scipy.signal import butter, filtfilt
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks
import scipy
import numpy as np
import pywt


fs = 1000.0
nyquist_rate = 0.5 * fs

order = 2
low = 1.0 / nyquist_rate
high = 40.0 / nyquist_rate


def load_signal(signal_path):
    signal, _ = wfdb.rdsamp(signal_path, channels=[1])
    return signal[96000:120012]


def filter_signal(signal):
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal[:, 0])


def extract_r_peaks(filtered_signal):
    r_peaks, _ = find_peaks(filtered_signal, distance=550)
    return r_peaks


def extract_s_indices(filtered_signal, r_peaks):
    s_point_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.04 * fs)
        local_min_index = np.argmin(filtered_signal[i:i + window_size])
        s_point_indices.append(i + local_min_index)
    return s_point_indices


def extract_q_indices(filtered_signal, r_peaks):
    q_point_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.1 * fs)
        start = i - window_size
        if start < 0:
            start = 0
        local_min_index = np.argmin(filtered_signal[start:i - 1])
        q_point_indices.append(start + local_min_index)
    return q_point_indices


def extract_t_peak(filtered_signal, r_peaks):
    t_peaks_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.4 * fs)
        local_max_index = np.argmax(filtered_signal[i + 40:i + 40 + window_size])
        t_peaks_indices.append(i + 40 + local_max_index)
    return t_peaks_indices


def extract_p_peak(filtered_signal, r_peaks):
    p_peaks_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.2 * fs)
        start = i - window_size - 40
        if start < 0:
            start = 0
        local_max_index = np.argmax(filtered_signal[start:i - 40])
        p_peaks_indices.append(start + local_max_index)
    return p_peaks_indices


def get_fiducial_points(signal):
    R_indices = extract_r_peaks(signal)
    return R_indices, extract_s_indices(signal, R_indices), extract_q_indices(signal, R_indices), \
        extract_t_peak(signal, R_indices), extract_p_peak(signal, R_indices)


def extract_fiducial_features(signal):
    R_indices, S_indices, Q_indices, T_indices, P_indices = get_fiducial_points(signal)
    features = []
    for j in range(len(P_indices)):
        QT_duration = (T_indices[j] - Q_indices[j]) / fs
        PQ_duration = ((Q_indices[j] - P_indices[j]) / fs) / QT_duration
        PR_duration = ((R_indices[j] - P_indices[j]) / fs) / QT_duration
        PS_duration = ((S_indices[j] - P_indices[j]) / fs) / QT_duration
        PT_duration = ((T_indices[j] - P_indices[j]) / fs) / QT_duration
        QS_duration = ((S_indices[j] - Q_indices[j]) / fs) / QT_duration
        QR_duration = ((R_indices[j] - Q_indices[j]) / fs) / QT_duration
        RS_duration = ((S_indices[j] - R_indices[j]) / fs) / QT_duration
        RT_duration = ((T_indices[j] - R_indices[j]) / fs) / QT_duration
        RP_freq = (signal[R_indices[j]] - signal[P_indices[j]])
        RT_freq = (signal[R_indices[j]] - signal[T_indices[j]])
        TP_freq = (signal[T_indices[j]] - signal[P_indices[j]])
        heart_beat_features = [QT_duration, PQ_duration, PR_duration, PS_duration,
                               PT_duration, QS_duration, QR_duration, RS_duration, RT_duration,
                               RP_freq, RT_freq, TP_freq]
        features.append(heart_beat_features)
    return np.array(features)
