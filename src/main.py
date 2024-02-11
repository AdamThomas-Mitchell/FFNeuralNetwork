import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


# utility fns - tidy later
def vectorise_y(labels):
    target_arr = np.zeros(shape=(len(labels), 10), dtype=float)
    for i, label in enumerate(labels):
        target_arr[i, int(label)] = int(1)

    return target_arr


# prepare data
digits = load_digits()

scaler = StandardScaler()
X = scaler.fit_transform(digits.data)
y = vectorise_y(digits.target)


### NETWORK ###

NUM_HIDDEN_LAYERS: int = 1
NUM_NODES: int = 30
