import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

fname = sys.argv[1]
data = pickle.load(open(fname, 'rb'))

ens_predictions = []
node_predictions = []
obs_age = []
for key, value in data.items():
	obs_age.append(key)
	ens_predictions.append(value[0])
	node_predictions.append(value[1])


plt.figure()
plt.plot(obs_age, ens_predictions)
plt.title("Ensemble Predictions")

plt.figure()
plt.plot(obs_age, node_predictions)
plt.title("Node Predictions")

plt.show()
