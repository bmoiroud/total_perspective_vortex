import mne
import joblib
import pygame
import time
import re

import numpy as np

from sys import argv
from sklearn.svm import NuSVC

from CSP import CSP
from train import get_features

labels = ["repos", "mouvement réel de la main gauche", "mouvement réel de la main droite", "mouvement imaginé de la main gauche", "mouvement imaginé de la main droite", \
			"mouvement réel des deux mains", "mouvement réel des deux pieds", "mouvement imaginé des deux mains", "mouvement imaginé des deux pieds"]

def		predict(data, sfreq, y, t, t1):
	data2 = mne.filter.notch_filter(np.concatenate(data, axis=1), sfreq, 60, verbose=50, method='iir')
	data2 = mne.filter.filter_data(data2, sfreq, 8, None, verbose=50, method='iir')
	res = svm.predict(csp.transform(get_features([data2], None, plot=False)))
	print("tps exec: {0:.5f}\ttps pred: {1:.5f}\tdétecté: {2:<36}\tcible: {3}".format(time.time()- t1, time.time() - t, labels[res[0]], labels[y]))

if __name__ == "__main__":
	if (len(argv) != 3):
		print("usage: python3 ./predict.py <edf_file> <model_file>")
		exit(-1)
	ch = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', \
		'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'Po7.', 'Po3.', \
		'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
	clock = pygame.time.Clock()

	try:
		raw = mne.io.read_raw_edf(argv[1], preload=True)
	except:
		print("Invalid edf file")
		exit(-1)
	events_list, _ = mne.events_from_annotations(raw, verbose=False)
	picks = mne.pick_channels(raw.info['ch_names'], ch)
	sfreq = raw.info['sfreq']

	n_samples = int(sfreq / 2)
	m = events_list[2][0] - events_list[1][0]
	n = int(re.findall(r'\d+', argv[1])[-1]) - 3

	try:
		svm = joblib.load(argv[2])
		csp = CSP()
		csp.load(re.findall(r'\d+', argv[2])[0] + ".npy")
	except:
		print("\nInvalid model")
		exit(-1)
	if (not isinstance(svm, NuSVC)):
		print("\nInvalid model")
		exit(-1)

	t = sfreq / n_samples

	data = []
	i = 0
	j = 0
	t1 = time.time()
	t2= t1
	while(i < raw.n_times):
		data.append(raw._data[picks, i : i + n_samples])
		if (len(data) * n_samples >= 300):
			if (j < 29 and events_list[j + 1][0] < i):
				j += 1
			if (events_list[j][2] != 1):
				event_id = events_list[j][2] + ((n % 4) * 2 - 1)
			else:
				event_id = 0

			predict(data, sfreq, event_id, t1, t2)
			t1 = time.time()
			if (len(data) * n_samples > 600):
				data.pop(0)
		i += n_samples
		clock.tick(t)