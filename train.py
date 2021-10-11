import os
import mne
import math
import joblib
import warnings

import numpy as np
import matplotlib.pyplot as plt

from CSP import CSP

from sys import argv
from sklearn.svm import NuSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def		baseline(data):
	m = np.mean(data, axis=1).reshape((-1, 1))
	return (data - m)

def		get_events(raw):
	labels = []
	events = []

	for i in range(len(raw)):
		base = (i % 4) * 2 - 1
		events_list, _ = mne.events_from_annotations(raw[i], verbose=False)
		interv = events_list[2][0] - events_list[1][0]
		start, stop = 0, 0
		
		for j in range(len(events_list)):
			start = events_list[j][0]
			stop = start + interv
			tmp = baseline(np.array(raw[i][:, start:stop][0]))
			events.append(tmp)
			if (events_list[j][2] != 1):
				labels.append(base + events_list[j][2])
			else:
				labels.append(0)
	x = np.arange(0, len(events[0][0])/raw[0].info['sfreq'], 1/raw[0].info['sfreq'])
	[plt.plot(x, events[0][i] - i * 1e-4) for i in range(len(events[0]))]
	plt.legend(raw[0].ch_names)
	plt.show()
	return (events, labels)

def		load_data(path, subject):
	ch = ['Fpz.', 'Fp1.', 'Fp2.', 'Afz.', 'Af3.', 'Af4.', 'Af7.', 'Af8.', 'Fz..', 'F1..', 'F2..', 'F3..', 'F4..', 'F5..',\
			'F6..', 'F7..', 'F8..', 'Ft7.', 'Ft8.', 'Pz..', 'P1..', 'P2..', 'P3..', 'P4..', 'P5..', 'P6..', 'P7..', 'P8..']
	raw = [mne.io.read_raw_edf("{0}/S{1:03d}/S{1:03d}R{2:02d}.edf".format(path, subject, run), preload=True) for run in range(3, 15)]
	raw[0].plot(block=True)
	
	[r.drop_channels(ch) for r in raw]
	[r.notch_filter(60, method='iir') for r in raw]
	[r.filter(8, None, picks=mne.pick_types(r.info, eeg=True), method='iir') for r in raw]
	
	raw[0].plot(block=True)
	events, labels = get_events(raw)
	return (events, labels, raw[0].info['sfreq'])

def		plot_fft(data, sfreq):
	data = np.fft.fft(data)
	N = len(data)
	f = np.linspace(0, sfreq, N)
	plt.bar(f[:N // 2], np.abs(data)[:N // 2] / N, 0.15)
	plt.show()

def		get_features(events, sfreq, plot=True):
	N = len(events[0][0]) // 2
	features = []

	if (plot):
		plot_fft(events[0][0], sfreq)
	for e in events:
		tmp = np.array([np.fft.fft(ch) for ch in e])
		tmp2 = []
		for i in range(tmp.shape[0]):
			tmp2.append(np.array([tmp[i].real[:N], tmp[i].imag[:N]]).reshape((N * 2)))
		features.append(np.array(tmp2).reshape((tmp.shape[0], N * 2)))
	return (np.array(features))

def		check_dataset(path, subject):
	for i in range(3, 15):
		if (os.path.exists("{0}/S{1:03d}/S{1:03d}R{2:02d}.edf".format(path, subject, i)) == False):
			return (False)
	return (True)

if __name__ == "__main__":
	if (len(argv) < 2):
		print("usage: python3 ./train.py <data_path> <subject_id>")
		exit(-1)
	if (check_dataset(argv[1], int(argv[2])) == False):
		print("Incomplete dataset")
		exit(-1)
	events, labels, sfreq = load_data(argv[1], int(argv[2]))
	features = get_features(events, sfreq)
	labels = np.array(labels)

	csp = CSP()
	cv = ShuffleSplit(200, 0.2, 0.8, random_state=0)

	svm = NuSVC(nu=0.11, gamma='scale', kernel='poly', degree=3)
	pipeline = Pipeline(steps=[('csp', csp), ('svm', svm)])

	score = cross_val_score(pipeline, features, labels, cv=cv, n_jobs=-1, verbose=10)
	print("\nscore moyen: " + str(score.mean()))
	
	csp.fit(features, labels)
	svm.fit(csp.transform(features), labels)
	
	csp.save("./{0}".format(argv[2]))
	joblib.dump(svm, "./model{0}".format(argv[2]))
