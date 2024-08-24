import wfdb
import os
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from joblib import dump

import numpy as np
import matplotlib.pyplot as plt

data_path = 'Data/'

ecg_signals =[]
ecg_fields = []

for subject in os.listdir(data_path):
    for x in os.listdir(data_path + subject):
        print(data_path + subject + '\\' + x.split('.')[0])
        signal, fields = wfdb.rdsamp(data_path + subject + '\\' + x.split('.')[0], channels=[1])
        ecg_signals.append(signal)
        ecg_fields.append(fields)
        break

print(len(ecg_signals[0]))
print(ecg_fields[0])

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
# Plot each ECG signal on a separate subplot
for i in range(4):
    row = i // 2
    col = i % 2
    axs[row, col].plot(ecg_signals[i])
    axs[row, col].set_xlabel('Sample number')
    axs[row, col].set_ylabel('Signal amplitude')
    axs[row, col].set_title(f'Signal {i+1}')

# Adjust subplot spacing and display the plot
plt.tight_layout()
plt.show()

start_sec = 0
end_sec = 2

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

for i in range(4):
    fs = ecg_fields[i]['fs']
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)
    row = i // 2
    col = i % 2

    axs[row, col].plot(ecg_signals[i])
    axs[row, col].set_xlabel('Sample number')
    axs[row, col].set_ylabel('Signal amplitude')
    axs[row, col].set_title(f'Signal {i + 1}')
    axs[row, col].set_xlim([start_sample, end_sample])

plt.show()

filtered_signals = []

# Define bandpass filter parameters
low_cut = 1.0
high_cut = 40.0
order = 2

# Apply bandpass filter to the signal
for i in range(4):
    fs = ecg_fields[i]['fs']
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg_signals[i][:, 0])
    filtered_signals.append(filtered)


start_sec = 0
end_sec = 2

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

for i in range(4):
    fs = ecg_fields[i]['fs']
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)
    row = i // 2
    col = i % 2

    axs[row, col].plot(filtered_signals[i])
    axs[row, col].set_xlabel('Sample number')
    axs[row, col].set_ylabel('Signal amplitude')
    axs[row, col].set_title(f'Signal {i + 1} bandpass')
    axs[row, col].set_xlim([start_sample, end_sample])

plt.show()

# Remove DC component
dc_removed_squared_signals = [(signal - np.mean(signal))**2 for signal in filtered_signals]

start_sec = 0
end_sec = 2

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

for i in range(4):
    fs = ecg_fields[i]['fs']
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)
    row = i // 2
    col = i % 2

    axs[row, col].plot(dc_removed_squared_signals[i])
    axs[row, col].set_xlabel('Sample number')
    axs[row, col].set_ylabel('Signal amplitude')
    axs[row, col].set_title(f'Signal {i + 1} bandpass & dc & sqr')
    axs[row, col].set_xlim([start_sample, end_sample])

plt.show()

# Normalize the squared signals
normalized_squared_signals = [signal / np.max(signal) for signal in dc_removed_squared_signals]


start_sec = 0
end_sec = 2

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

for i in range(4):
    fs = ecg_fields[i]['fs']
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)
    row = i // 2
    col = i % 2

    axs[row, col].plot(normalized_squared_signals[i])
    axs[row, col].set_xlabel('Sample number')
    axs[row, col].set_ylabel('Signal amplitude')
    axs[row, col].set_title(f'Signal {i + 1} bandpass & dc & sqr & norm')
    axs[row, col].set_xlim([start_sample, end_sample])

plt.show()


R_indices = []
for i in range (4):
    peaks, _ = find_peaks(filtered_signals[i], distance=550)
    R_indices.append(peaks)

S_indices = []
fs = 1000
for i in range(4):
    s_idx = []
    for r_peak in R_indices[i]:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.04 * fs)
        local_min_index = np.argmin(filtered_signals[i][r_peak:r_peak+window_size])
        s_idx.append(r_peak + local_min_index)
    S_indices.append(s_idx)

Q_indices = []
fs = 1000
for i in range(4):
    q_idx = []
    for r_peak in R_indices[i]:
        window_size = int(0.1 * fs)
        start = r_peak - window_size
        if start < 0:
            start = 0
        local_min_index = np.argmin(filtered_signals[i][start:r_peak -1])
        q_idx.append(start + local_min_index)
    Q_indices.append(q_idx)


T_indices = []
fs = 1000
for i in range(4):
    t_idx = []
    for r_peak in R_indices[i]:
        window_size = int(0.4 * fs)
        local_max_index = np.argmax(filtered_signals[i][r_peak + 40:r_peak+ 40 + window_size])
        t_idx.append(r_peak + 40 + local_max_index)
    T_indices.append(t_idx)

P_indices = []
fs = 1000
for i in range(4):
    p_idx = []
    for r_peak in R_indices[i]:
        window_size = int(0.2 * fs)
        start = r_peak - window_size - 40
        if start < 0:
            start = 0
        local_max_index = np.argmax(filtered_signals[i][start:r_peak -40])
        p_idx.append(start + local_max_index)
    P_indices.append(p_idx)

def pan_and_tompkins(signals):
    convolved_Signals = []
    for i in signals:
        dx = np.diff(i)
        dx = np.square(dx)
        dx = np.convolve(dx, np.ones(200), mode='same')
        convolved_Signals.append(dx) #moving window integration
        plt.plot(dx[0:3000])
        plt.show()
    return convolved_Signals

def find_qrs_onset_and_offset(r_peaks, convolved_signals):
    qrs_onsets = []
    qrs_offsets = []

    for r_peak in r_peaks:
        # Find QRS onset
        qrs_onset = r_peak
        while qrs_onset > 0 and convolved_signals[qrs_onset] > 0.1 * convolved_signals[r_peak]:
            qrs_onset -= 1

        # Find QRS offset
        qrs_offset = r_peak
        while qrs_offset < len(convolved_signals)-1 and convolved_signals[qrs_offset] > 0.1 *convolved_signals[r_peak]:
            qrs_offset += 1

        qrs_onsets.append(qrs_onset)
        qrs_offsets.append(qrs_offset)
    return qrs_onsets, qrs_offsets

convolved_signals = pan_and_tompkins(filtered_signals)
QRS_onset = []
QRS_offset = []
for i in range(4):
    on, off = find_qrs_onset_and_offset(R_indices[i], convolved_signals[i])
    QRS_onset.append(on)
    QRS_offset.append(off)

def find_p_wave_onset_and_offset(peak_indices, convolved_signals):
    p_onsets = []
    p_offsets = []

    for peak_index in peak_indices:
        # Find P wave onset
        p_onset = peak_index
        while p_onset > 0 and convolved_signals[p_onset] > 0.1 * convolved_signals[peak_index]:
            p_onset -= 1

        # Find P wave offset
        p_offset = peak_index
        while p_offset < len(convolved_signals) - 1 and convolved_signals[p_offset] > 0.1 * convolved_signals[peak_index]:
            p_offset += 1

        p_onsets.append(p_onset)
        p_offsets.append(p_offset)

    return p_onsets, p_offsets

def find_t_wave_onset_and_offset(peak_indices, convolved_signals):
    t_onsets = []
    t_offsets = []

    for peak_index in peak_indices:
        # Find T wave onset
        t_onset = peak_index
        while t_onset > 0 and convolved_signals[t_onset-1] > 0.1 * convolved_signals[peak_index-1]:
            t_onset -= 1

        # Find T wave offset
        t_offset = peak_index
        while t_offset < len(convolved_signals) - 1 and convolved_signals[t_offset] > 0.1 * convolved_signals[peak_index]:
            t_offset += 1

        t_onsets.append(t_onset)
        t_offsets.append(t_offset)

    return t_onsets, t_offsets

P_onset = []
P_offset = []
T_onset = []
T_offset = []

for i in range(4):
    p_on, p_off = find_p_wave_onset_and_offset(P_indices[i], convolved_signals[i])  # Assuming P_peak_indices are available
    P_onset.append(p_on)
    P_offset.append(p_off)

    t_on, t_off = find_t_wave_onset_and_offset(T_indices[i], convolved_signals[i])  # Assuming T_peak_indices are available
    T_onset.append(t_on)
    T_offset.append(t_off)






for i in range(4):
    plt.plot(filtered_signals[i])
    plt.scatter(R_indices[i], filtered_signals[i][R_indices[i]], c='red')
    plt.scatter(Q_indices[i], filtered_signals[i][Q_indices[i]], c='green')
    plt.scatter(S_indices[i], filtered_signals[i][S_indices[i]], c='blue')
    plt.scatter(T_indices[i], filtered_signals[i][T_indices[i]], c='cyan')
    plt.scatter(P_indices[i], filtered_signals[i][P_indices[i]], c='magenta')
    plt.scatter(P_onset[i], filtered_signals[i][P_onset[i]], c='yellow')
    plt.scatter(P_offset[i], filtered_signals[i][P_offset[i]], c='orange')
    plt.scatter(T_onset[i], filtered_signals[i][T_onset[i]], c='purple')
    plt.scatter(T_offset[i], filtered_signals[i][T_offset[i]], c='brown')
    plt.scatter(QRS_onset[i], filtered_signals[i][QRS_onset[i]], c='pink')
    plt.scatter(QRS_offset[i], filtered_signals[i][QRS_offset[i]], c='gray')

    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.xlim(0, 1500)
    plt.title('ECG Signal with Fiducial Points')
    plt.show()

    features3 = []
    beats_classes = []
    fs = 1000.0

    for i in range(4):
        for j in range(len(P_indices[i])):
            QT_duration = (T_indices[i][j] - Q_indices[i][j]) / fs
            PQ_duration = ((Q_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            PR_duration = ((R_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            PS_duration = ((S_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            PT_duration = ((T_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            QS_duration = ((S_indices[i][j] - Q_indices[i][j]) / fs) / QT_duration
            QR_duration = ((R_indices[i][j] - Q_indices[i][j]) / fs) / QT_duration
            RS_duration = ((S_indices[i][j] - R_indices[i][j]) / fs) / QT_duration
            RT_duration = ((T_indices[i][j] - R_indices[i][j]) / fs) / QT_duration
            RP_freq = (filtered_signals[i][R_indices[i][j]] - filtered_signals[i][P_indices[i][j]])
            RT_freq = (filtered_signals[i][R_indices[i][j]] - filtered_signals[i][T_indices[i][j]])
            TP_freq = (filtered_signals[i][T_indices[i][j]] - filtered_signals[i][P_indices[i][j]])
            heart_beat_features = [QT_duration, PQ_duration, PR_duration, PS_duration,
                                   PT_duration, QS_duration, QR_duration, RS_duration, RT_duration,
                                   RP_freq, RT_freq, TP_freq]
            features3.append(heart_beat_features)
            beats_classes.append(i)

    print(len(features3))
    features3 = np.array(features3)
    beats_classes = np.array(beats_classes)

    print(features3.shape)
    print(beats_classes.shape)


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(features3, beats_classes, test_size=0.2, shuffle=True)

# Logistic Regression
logistic_classifier = LogisticRegression(solver="liblinear")
logistic_classifier.fit(X_train, y_train)

# Save logistic regression model
dump(logistic_classifier, 'Models/LogisticRegression_classifier.joblib')

# Predict with logistic regression
y_pred_logistic = logistic_classifier.predict(X_test)
acc_logistic = metrics.accuracy_score(y_test, y_pred_logistic)

# SVM Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Save SVM model
dump(svm_classifier, 'Models/SVM_classifier.joblib')

# Predict with SVM
y_pred_svm = svm_classifier.predict(X_test)
acc_svm = metrics.accuracy_score(y_test, y_pred_svm)

# Random Forest Classifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, y_train)

# Save Random Forest model
dump(random_forest_classifier, 'Models/RandomForest_classifier.joblib')

# Predict with Random Forest
y_pred_rf = random_forest_classifier.predict(X_test)
acc_rf = metrics.accuracy_score(y_test, y_pred_rf)

# Naive Bayes Classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

# Save Naive Bayes model
dump(naive_bayes_classifier, 'Models/NaiveBayes_classifier.joblib')

# Predict with Naive Bayes
y_pred_nb = naive_bayes_classifier.predict(X_test)
acc_nb = metrics.accuracy_score(y_test, y_pred_nb)

# Print accuracies
print("Logistic Regression Accuracy:", acc_logistic)
print("SVM Accuracy:", acc_svm)
print("Random Forest Accuracy:", acc_rf)
print("Naive Bayes Accuracy:", acc_nb)

