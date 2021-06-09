#! /usr/bin/python3

import csv
import collections
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from hmmlearn import hmm
from sklearn.naive_bayes import MultinomialNB

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
		for j in range(1,4):
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
	return np.array(Labels, dtype=np.int32).reshape(-1, 1)
	
def shorten(Labels):
	result = list()
	for label in Labels:
		if not result or result[-1] != label:
			result.append(label)
	return np.array(result, dtype=np.int32).reshape(-1, 1)

def resample(Labels, Length):
	"""Simple resampling of the Label list to target Length"""
	result = [ 0 for _ in range(Length) ]
	dist = float(len(Labels))/float(Length)
	for k in range(Length):
		result[k] = Labels[int(k * dist)]
	return np.array(result, dtype=np.int32)

def make_multinomial(km):
	"""The resampled labeled data or the discretized data may not be
	following a multinomial distribution. Here, we try to fix it."""
	pass

Filenames = [ '1557913740208.csv' ]

Data = [ read_motion(filename, 1.0) for filename in Filenames ]
dfa = pandas.DataFrame(Data[0], columns=['time', 'ax', 'ay', 'az'])
seaborn.relplot(kind="line", data=dfa[["ax", "ay", "az"]])

for Acc in Data:
	idle_filter(Acc, 0.5)
	dir_equiv_filter(Acc, 0.5)

dfa = pandas.DataFrame(np.concatenate(Data), columns=['time', 'x', 'y', 'z'])

km = KMeans(n_clusters=14)
km.fit(dfa[["x", "y", "z"]])

print(km.cluster_centers_)
print()
print("Fitting HMM")
model = hmm.MultinomialHMM(n_components=8)
Training = [label(Acc, km.cluster_centers_) for Acc in Data]
model.fit(np.concatenate(Training), [len(X) for X in Training])

print("Fitting naive Bayes")
Waving = [ model.predict(resample(t, 40)).reshape(1, -1) for t in Training]
Other = [ np.random.randint(14, size=(1, 40)) for _ in range(20) ]

clf = MultinomialNB()
Observations = np.concatenate(Waving + Other )
Targets = np.array([1 for _ in range(len(Waving))] + [2 for _ in range(len(Other))])
clf.fit(Observations, Targets)

input("Press Enter to detect.")

# Read a gesture from st.csv, filter it, label it, match it to the
# HMM and classify it
Acc2 = read_motion(Filenames[0])
idle_filter(Acc2)
dir_equiv_filter(Acc2)
stateseq = model.predict(resample(label(Acc2, km.cluster_centers_), 40))
#stateseq = np.random.randint(14, size=(1, 40))
cls = clf.predict_proba(stateseq.reshape(1, -1))
print(cls)
stateseq = np.random.randint(14, size=(1, 40))
cls = clf.predict_proba(stateseq.reshape(1, -1))
print(cls)

input("Press Enter to exit.")
sys.exit(0)
