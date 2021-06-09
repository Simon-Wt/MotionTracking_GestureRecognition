from get_rotation_matrix import getRotationMatrix
import matplotlib.pyplot as plt
import csv
import statistics
import pandas
import seaborn

import numpy as np

counter = 0
a1, a2, a3 = [], [], []
g1, g2, g3 = [], [], []
m1, m2, m3 = [], [], []
p = []

total = 13030

segments = [(list(), list(), list(), list()),
            (list(), list(), list(), list()),
            (list(), list(), list(), list()),
            (list(), list(), list(), list())]

Acc = list()
Grav = list()
Mag = list()
Rot = list()

with open("Tag2/5-5-5-5.csv") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=";")
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

    # First Transition HAND UP
    Accs = list()
    Gravs = list()
    Mags = list()
    Rots = list()
    segCount = 4
    segSize = int(len(Acc)/segCount)
    for i in range(1, segCount):
        Accs.append(Acc[segSize*(i-1):segSize*i])
        Gravs.append(Grav[segSize*(i-1):segSize*i])
        Mags.append(Mag[segSize*(i-1):segSize*i])
        Rots.append(Rots[segSize*(i-1):segSize*i])

    dfp = pandas.DataFrame(Acc, columns=['time', 'x', 'y', 'z'])
    seaborn.relplot(kind="line", data=dfp[["x", "y", "z"]])
    plt.show()
