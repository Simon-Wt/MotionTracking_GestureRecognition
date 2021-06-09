#! /usr/bin/python3

import csv
import collections
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from get_rotation_matrix import getRotationMatrix


def read_motion(filename, rotate=True, alpha=None):
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
            if row[1] == '2':  # Linear acceleration
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
    """Return a list of labels for the accelation vector list Acc by
    looking for the closest entry in Points, which are assumed to be the
    cluster centres."""
    Labels = [0 for _ in range(len(Acc))]
    for i in range(len(Acc)):
        Q, dQ = None, sys.float_info.max
        for p in range(len(Points)):
            t = Acc[i][1:] - np.array(Points[p])
            d = np.sqrt(sum(t*t))
            if d < dQ:
                Q, dQ = p, d
        Labels[i] = Q
    return Labels


# filename = '/csv/5-5-5-5.csv'
# filename = '/csv/circle_4sec.csv'
filename = '/csv/swingingLong.csv'
# filename = '/csv/swingingShort.csv'
Acc = read_motion(filename)
dfa = pandas.DataFrame(Acc, columns=['time', 'x', 'y', 'z'])
seaborn.relplot(kind="line", data=dfa[["x", "y", "z"]])

# integrate acceleration to get speed
Spd = np.zeros(dfa[["x", "y", "z"]].shape)
for i in range(1, len(Acc)):
    dT = (Acc[i][0] - Acc[i-1][0]) / 1000.0  # elapsed time in milliseconds
    Spd[i] = Spd[i - 1] + dT * Acc[i-1][1:]

dfv = pandas.DataFrame(Spd, columns=['x', 'y', 'z'])
seaborn.relplot(kind="line", data=dfv)

# integrate speed to get position
Pos = np.zeros(dfv.shape)
for i in range(1, len(Acc)):
    dT = (Acc[i][0] - Acc[i-1][0]) / 1000.0  # elapsed time in milliseconds
    Pos[i] = Pos[i - 1] + dT * Spd[i-1]

dfp = pandas.DataFrame(Pos, columns=['x', 'y', 'z'])
seaborn.relplot(kind="line", data=dfp)


def KMeansGraph(dataframe, k, name, printToTerminal=False):
    kma = KMeans(n_clusters=k)
    kma.fit(dataframe[["x", "y", "z"]])

    if(printToTerminal):
        print(name + " k=" + k)
        print(kma.cluster_centers_)
        print(kma.labels_)

    Quantized = [kma.cluster_centers_[i] for i in kma.labels_]
    df = pandas.DataFrame(Quantized, columns=['x', 'y', 'z'])

    seaborn.relplot(kind="line", height=4, data=df).set_axis_labels(
        "time", "value").set(title=name).tight_layout()

    return 0


for i in (2, 4, 8, 16):
    KMeansGraph(dfa, i, "KMeans Acceleration k=" + str(i))
    KMeansGraph(dfv, i, "KMeans Speed k=" + str(i))
    KMeansGraph(dfp, i, "KMeans Position k=" + str(i))

plt.savefig("fig.png")
plt.show()

input("Press Enter to exit.")
sys.exit(0)
