#! /usr/bin/python3

import csv
import collections
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

# from peakdetect import peakdetect

def read_motion(filename, rotate=True, alpha=1.0/12.0):
	"""Read and preprocess the motion file."""
	Acc = list()
	Rot = list()

	# Read data
	with open(filename) as csvfile:
		csvreader = csv.reader(csvfile, delimiter=';')
		for row in csvreader:
			if row[1] == '2':
				a = np.array([float(row[0]), float(row[2]), float(row[3]), float(row[4])])
				Acc.append(a)
			elif row[1] == '7':
				m = np.matrix([[float(row[2]), float(row[3]), float(row[4])],
							   [float(row[5]), float(row[6]), float(row[7])],
							   [float(row[8]), float(row[9]), float(row[10])]])
				Rot.append([float(row[0]), m])
            
	# Rotate acceleration vectors
	if rotate:
		k=0
		for i in range(len(Acc)):
			while k < len(Rot) and Acc[k][0] >= Rot[k][0]:
				k = k + 1
			av = np.array(Acc[i][1:])
			# a = av.dot(Rot[k][1])
			a = (Rot[k][1]).dot(av)
			for j in range(1,4):
				Acc[i][j] = a[0, j-1]

	if alpha:# Exponentially fading filter
		for i in range(1, len(Acc)):
			ts = Acc[i][0] # remember timestamp, we do not want to filter it
			Acc[i] = Acc[i] * alpha + Acc[i-1] * (1.0-alpha)
			Acc[i][0] = ts
	return Acc

def label(Acc, Points):
	Labels = [ 0 for _ in range(len(Acc))]
	for i in range(len(Acc)):
		Q, dQ = None, 10.0
		for p in range(len(Points)):
			t = Acc[i][1:] - np.array(Points[p])
			d = np.sqrt(sum(t*t))
			if d < dQ:
				Q, dQ = p, d
		for j in range(1,4):
			Labels[i] = Q
	return Labels

Acc = read_motion('1557913740208.csv')	
dfa = pandas.DataFrame(Acc, columns=['time', 'ax', 'ay', 'az'])

dfa2 = dfa.copy()
dfa2.drop('time', axis=1, inplace=True)
seaborn.relplot(kind="line", data=dfa2)

km = KMeans(n_clusters=8)
km.fit(dfa[["ax", "ay", "az"]])

QAcc = [ km.cluster_centers_[i] for i in km.labels_]
dfq = pandas.DataFrame(QAcc, columns=['ax', 'ay', 'az'])

seaborn.relplot(kind="line", data=dfq)

# integrate acceleration to get speed
Spd = np.zeros(dfa.shape)
Spd[0, 0] = Acc[0][0]   # copy time stamp
for i in range(1, len(Acc)):
    dT = (Acc[i][0] - Acc[i-1][0]) / 1000.0 # elapsed time in milliseconds
    Spd[i, 0] = dfa["time"][i] # Copy time stamp
    for j in range(1,4):
        Spd[i, j] = Spd[i - 1, j] + dT * Acc[i-1][j]

dfv = pandas.DataFrame(Spd, columns=['time', 'vx', 'vy', 'vz'])
seaborn.relplot(kind="line", data=dfv[['vx', 'vy', 'vz']])

# integrate speed to get position
Pos = np.zeros(dfv.shape)
Pos[0, 0] = Spd[0,0]   # copy time stamp
for i in range(1, len(Acc)):
    dT = (Spd[i,0] - Spd[i-1,0]) / 1000.0 # elapsed time in milliseconds
    Pos[i, 0] = Spd[i,0] # Copy time stamp
    for j in range(1,4):
        Pos[i, j] = Pos[i - 1, j] + dT * Spd[i-1,j]

dfp = pandas.DataFrame(Pos, columns=['time', 'x', 'y', 'z'])
seaborn.relplot(kind="line", data=dfp[['x', 'y', 'z']])
plt.show()

input("press key")
Acc2 = read_motion('1557213040059.csv')

Labels = label(Acc2, km.cluster_centers_)	
print(Labels)
print(km.labels_ == Labels)
sys.exit(0)

