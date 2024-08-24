import numpy as np
from joblib import load

from HCISingleSignals import filter_signal
from HCISingleSignals import extract_fiducial_features


def get_features_and_classifier(signal):
    filtered_signal = filter_signal(signal)
    features = extract_fiducial_features(filtered_signal)
    clf = load('../Models/RandomForest_classifier.joblib')

    return features, clf


def evaluate_authentication(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    half_length = len(predictions) / 2
    for value, count in zip(unique, counts):
        if count > half_length:
            return value
    return -1


def run(signal):
    features, clf = get_features_and_classifier(signal)
    predictions = clf.predict(features)
    authenticated_person_index = evaluate_authentication(predictions)
    return authenticated_person_index
def similarity(signal):
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    maxi = 0
    features, clf = get_features_and_classifier(signal)
    predictions = clf.predict(features)
    print(predictions)
    for i in predictions:
        if i == 0:
            count0 += 1
        if i == 1:
            count1 += 1
        if i == 2:
            count2 += 1
        if i == 3:
            count3 += 1

    maxi = max(count0, count1, count2, count3)
    similarity = maxi / (count0 + count2 + count3 + count1) * 100
    print(similarity)
    return similarity
