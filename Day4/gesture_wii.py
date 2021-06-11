#! /usr/bin/python3

import collections
import numpy as np
import pandas

import seaborn
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

import sys
import motion

from sklearn.cluster import KMeans
from hmmlearn import hmm
from sklearn.naive_bayes import MultinomialNB

# List all file names here. Use a sample per file.
Filenames = ["csv/my/5-5-5-5.csv"]

Alpha = 1.0/8.0
Delta = 0.5
Epsilon = 0.2
Length = 25
Clusters = 18
States = 10

# from peakdetect import peakdetect


def idle_filter(Acc, delta=0.5):
    """Set all vectors with an acceleration magnitude smaller than
    delta to 0"""
    for i in range(len(Acc)):
        d = np.sqrt(sum(Acc[i][1:]*Acc[i][1:]))
        if d < delta:
            Acc[i][1:] = [0.0, 0.0, 0.0]


def dir_equiv_filter(Acc, epsilon=0.2):
    """If two consecutive vectors are less than epsilon apart,
    treat them as equivalent"""
    for i in range(1, len(Acc)):
        equivalent = True
        for j in range(1, 4):
            d = np.abs(Acc[i-1][j]-Acc[i][j])
            if d >= epsilon:
                equivalent = False
                break
        if equivalent:
            Acc[i][1:] = Acc[i-1][1:]


def label(Acc, Points):
    Labels = [0 for _ in range(len(Acc))]
    for i in range(len(Acc)):
        Q, dQ = None, 10.0
        for p in range(len(Points)):
            t = Acc[i][1:] - np.array(Points[p])
            d = np.sqrt(sum(t*t))
            if d < dQ:
                Q, dQ = p, d
        Labels[i] = Q
    return np.array(Labels, dtype=np.int32).reshape(-1, 1)


def shorten(Labels):
    """Removes the duplicates."""
    result = list()
    for label in Labels:
        if not result or result[-1] != label:
            result.append(label)
    return np.array(result, dtype=np.int32).reshape(-1, 1)


def resample(Labels, Length):
    """Simple resampling of the Label list to target Length. Uses decimation."""
    result = [0 for _ in range(Length)]
    dist = np.float64(len(Labels))/np.float64(Length)
    for k in range(Length):
        result[k] = Labels[int(k * dist)]
    return np.array(result, dtype=np.int32)


def make_multinomial(km):
    """The resampled labeled data or the discretized data may not be
    following a multinomial distribution. Here, we try to fix it."""
    pass


def main():
    """The main function"""
    Data = [motion.read(filename, alpha=Alpha)
            for filename in Filenames]

    for Acc in Data:
        idle_filter(Acc, Delta)
        dir_equiv_filter(Acc, Epsilon)

    dfa = pandas.DataFrame(np.concatenate(Data), columns=[
                           'time', 'ax', 'ay', 'az'])

    print("KMeans clustering")
    km = KMeans(n_clusters=Clusters)
    km.fit(dfa[["ax", "ay", "az"]])

    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, elev=48, azim=134)
    ax.scatter(dfa["ax"], dfa["ay"], dfa["az"],
               c=km.labels_.astype(np.float64), edgecolor='w')
    fig.show()

    print(f'means = {km.cluster_centers_}\n\n')

    print("Fitting HMM")
    Training = [shorten(label(Acc, km.cluster_centers_)) for Acc in Data]

    model = hmm.MultinomialHMM(n_components=States)
    model.fit(np.concatenate(Training), [len(X) for X in Training])

    print(
        f'startprob = {model.startprob_}\ntransmat = {model.transmat_}\nemissionprob = {model.emissionprob_}\n\n')

    print("Fitting naive Bayes classifier")

    # We need random data to match what does not fit a gesture
    Other = [np.random.randint(14, size=(1, Length)) for _ in range(20)]
    Data = [model.predict(resample(t, Length).reshape(1, -1))
            for t in Training]

    clf = MultinomialNB(alpha=1.0, fit_prior=False)
    Observations = np.concatenate(Other + Data)
    # This lists the classes to match to.
    Targets = np.array([1 for _ in range(len(Other))] +
                       [2 for _ in range(len(Data))])
    clf.fit(Observations, Targets)

    if False:
        input("Press Enter to demonstrate.")
        Acc2 = motion.read(Filenames[0], alpha=Alpha)
        df = pandas.DataFrame(Acc2, columns=["time", "ax", "ay", "az"])
        seaborn.relplot(kind="line", data=df[["ax", "ay", "az"]])
        idle_filter(Acc2, Delta)
        df = pandas.DataFrame(Acc2, columns=["time", "ax", "ay", "az"])
        seaborn.relplot(kind="line", data=df[["ax", "ay", "az"]])
        dir_equiv_filter(Acc2, Epsilon)
        df = pandas.DataFrame(Acc2, columns=["time", "ax", "ay", "az"])
        seaborn.relplot(kind="line", data=df[["ax", "ay", "az"]])
        labelseq = resample(label(Acc2, km.cluster_centers_), Length)
        stateseq = model.predict(labelseq)
        print(stateseq)
        QAcc = (km.cluster_centers_[labelseq.reshape(1, -1)])[0]  # WTF?
        df = pandas.DataFrame(QAcc, columns=["ax", "ay", "az"])
        seaborn.relplot(kind="line", data=df)
        df = pandas.DataFrame(list(zip(labelseq, stateseq)),
                              columns=["Label", "States"])
        seaborn.relplot(data=df)
        cls = clf.predict_proba(stateseq.reshape(1, -1))
        print(cls)

    plt.show()

    input("Press Enter to detect.")

    # For true positive
    for f in Filenames:
        Acc2 = motion.read(f, alpha=Alpha)
        idle_filter(Acc2, Delta)
        dir_equiv_filter(Acc2, Epsilon)
        stateseq = model.predict(
            resample(label(Acc2, km.cluster_centers_), Length))
        cls = clf.predict_proba(stateseq.reshape(1, -1))
        print(
            f'File {f}:\n\tstateseq = {stateseq}\n\tclassification = {cls}\n\n')

    # For most likely negative
    stateseq = np.random.randint(14, size=(1, Length))
    print(stateseq)
    cls = clf.predict_proba(stateseq.reshape(1, -1))
    print(cls)

    input("Press Enter to exit.")
    sys.exit(0)


main()
