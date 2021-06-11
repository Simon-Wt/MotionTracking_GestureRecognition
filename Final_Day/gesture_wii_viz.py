#! /usr/bin/python3

import csv
import collections
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

import os

import sys
from sklearn.cluster import KMeans
from hmmlearn import hmm
from sklearn.naive_bayes import MultinomialNB
from get_rotation_matrix import getRotationMatrix

# one file per sample of one gesture (removing beginning and end touching beforehand)
folderPath = "csv/polish-left/"
Filenames = os.listdir(folderPath)
for i in range(0, len(Filenames)):
    Filenames[i] = folderPath + Filenames[i]
Filenames = ["csv/idle/idle_hand.csv"]
print("Filenames:", Filenames)
Alpha = 1.0/8.0
Delta = 1.5
Epsilon = 0.15
Length = 40
Clusters = 14
States = 8

np.random.seed(420)
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


def read_motion(filename, rotate=True, alpha=1.0/12.0):
    """Read and preprocess the motion file.

    A simple moving average filter with window size N can be
    approximated by an EMA with alpha=2/(N+1).
    """
    Acc = list()
    Grav = list()
    Mag = list()
    Rot = list()

    # Read data
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if row[1] == '2':
                a = np.array([float(row[0]), float(row[2]),
                             float(row[3]), float(row[4])])
                Acc.append(a)
            elif row[1] == '1':  # Gravity
                g = np.array([float(row[0]), float(row[2]),
                             float(row[3]), float(row[4])])
                Grav.append(g)
                if len(Mag) > 0:
                    Rot.append(
                        [float(row[0]), getRotationMatrix(Grav[-1], Mag[-1])])
            elif row[1] == '4':  # Magnetic field
                m = np.array([float(row[0]), float(row[2]),
                             float(row[3]), float(row[4])])
                Mag.append(m)
                if len(Grav) > 0:
                    Rot.append(
                        [float(row[0]), getRotationMatrix(Grav[-1], Mag[-1])])

    # Rotate acceleration vectors
    if rotate:
        k = 0
        for i in range(len(Acc)):
            while k < len(Rot) and k < len(Acc) and Acc[k][0] >= Rot[k][0]:
                k = k + 1
            av = np.array(Acc[i][1:])
            # a = av.dot(Rot[k][1])
            a = (Rot[k][1]).dot(av)
            for j in range(1, 4):
                Acc[i][j] = a[0, j-1]

    if alpha:  # Exponentially fading filter
        for i in range(1, len(Acc)):
            ts = Acc[i][0]  # remember timestamp, we do not want to filter it
            Acc[i] = Acc[i] * alpha + Acc[i-1] * (1.0-alpha)
            Acc[i][0] = ts
    return Acc


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
    """Removes the duplicates"""
    result = list()
    for label in Labels:
        if not result or result[-1] != label:
            result.append(label)
    return np.array(result, dtype=np.int32).reshape(-1, 1)


def resample(Labels, Length):
    """Simple resampling of the Label list to target Length"""
    result = [0 for _ in range(Length)]
    dist = float(len(Labels))/float(Length)
    for k in range(Length):
        result[k] = Labels[int(k * dist)]
    return np.array(result, dtype=np.int32)


def make_multinomial(km):
    """The resampled labeled data or the discretized data may not be
    following a multinomial distribution. Here, we try to fix it."""
    pass


Data = [read_motion(filename, alpha=Alpha) for filename in Filenames]
dfa = pandas.DataFrame(Data[0], columns=['time', 'ax', 'ay', 'az'])
seaborn.relplot(kind="line", data=dfa[["ax", "ay", "az"]])

for Acc in Data:
    idle_filter(Acc, Delta)
    dir_equiv_filter(Acc, Epsilon)

dfa = pandas.DataFrame(np.concatenate(Data), columns=[
                       'time', 'ax', 'ay', 'az'])

km = KMeans(n_clusters=Clusters)
km.fit(dfa[["ax", "ay", "az"]])

print("Cluster Center:")
print(km.cluster_centers_)
print()
print("Fitting HMM")
model = hmm.MultinomialHMM(n_components=States)
Training = [label(Acc, km.cluster_centers_) for Acc in Data]
model.fit(np.concatenate(Training), [len(X) for X in Training])
print("startprob = ", model.startprob_)
print("transmat = ", model.transmat_)
print("emissionprob = ", model.emissionprob_)

print("Fitting naive Bayes")
Other = [np.random.randint(14, size=(1, Length)) for _ in range(20)]
Waving = [model.predict(resample(t, Length)).reshape(1, -1) for t in Training]
Stirring = [np.random.randint(14, size=(1, Length))]

clf = MultinomialNB(alpha=1.0, fit_prior=False)  # Naive Bayes
Observations = np.concatenate(Other + Waving + Stirring)
Targets = np.array([1 for _ in range(len(Other))] +
                   [2 for _ in range(len(Waving))] + [3 for _ in range(len(Stirring))])
clf.fit(Observations, Targets)

input("Press Enter to demonstrate.")
fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, elev=48, azim=134, auto_add_to_figure=False)
ax.scatter(dfa["ax"], dfa["ay"], dfa["az"],
           c=km.labels_.astype(float), edgecolor='k')
fig.show()

filename = Filenames[0]
Acc2 = read_motion(Filenames[0], alpha=Alpha)
df = pandas.DataFrame(Acc2, columns=["time", "ax", "ay", "az"])
seaborn.relplot(kind="line", data=df[["ax", "ay", "az"]]).set(
    title=filename + " normal").tight_layout()

idle_filter(Acc2, Delta)
df = pandas.DataFrame(Acc2, columns=["time", "ax", "ay", "az"])
seaborn.relplot(kind="line", data=df[["ax", "ay", "az"]]).set(
    title=filename + " w/  idle Filter").tight_layout()

dir_equiv_filter(Acc2, Epsilon)
df = pandas.DataFrame(Acc2, columns=["time", "ax", "ay", "az"])
seaborn.relplot(kind="line", data=df[["ax", "ay", "az"]]).set(
    title=filename + " w/ idle Filter + dir equiv Filter").tight_layout()

labelseq = resample(label(Acc2, km.cluster_centers_), Length)
stateseq = model.predict(labelseq)
QAcc = (km.cluster_centers_[labelseq.reshape(1, -1)])[0]  # WTF?
df = pandas.DataFrame(QAcc, columns=["ax", "ay", "az"])
seaborn.relplot(kind="line", data=df).set(
    title=filename + " quantized after filtering").tight_layout()

df = pandas.DataFrame(list(zip(labelseq, stateseq)),
                      columns=["Label", "States"])
seaborn.relplot(data=df).set(
    title=filename + " Staterepresentation").tight_layout()

cls = clf.predict_proba(stateseq.reshape(1, -1))
print(cls)

input("Press Enter to detect.")

# For true positive
Acc2 = read_motion(Filenames[0], alpha=Alpha)
idle_filter(Acc2, Delta)
dir_equiv_filter(Acc2, Epsilon)
stateseq = model.predict(resample(label(Acc2, km.cluster_centers_), Length))
cls = clf.predict_proba(stateseq.reshape(1, -1))
print(cls)

# For most likely negative
stateseq = np.random.randint(14, size=(1, Length))
cls = clf.predict_proba(stateseq.reshape(1, -1))
print("Prediction of randomly generated state sequences. " +
      "The System is calculating the propability of being a gesture for each seq.")
print(cls)

plt.show()
input("Press Enter to exit.")
sys.exit(0)
