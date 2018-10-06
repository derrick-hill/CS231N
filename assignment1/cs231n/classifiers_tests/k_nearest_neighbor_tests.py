import numpy as np
import sys
import os

path_to_this_file = os.path.dirname(os.path.abspath(__file__))
path_to_cs231n = path_to_this_file  + "/../.."
path_to_datasets = path_to_cs231n + "/cs231n"
path_to_cifar10 =  path_to_datasets + "/datasets/cifar-10-batches-py"

# Make these paths work on both Windows and OSX.
norm_path_to_datasets = os.path.normpath(path_to_datasets)
norm_path_to_cs231n = os.path.normpath(path_to_cs231n)
norm_path_to_cifar10 = os.path.normpath(path_to_cifar10)

sys.path.append(norm_path_to_cs231n)
sys.path.append(norm_path_to_datasets)

# for p in sys.path:
#     print(p)

import cs231n.data_utils
from data_utils import load_CIFAR10
     
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

# Load the raw CIFAR-10 data.
X_train, y_train, X_test, y_test = load_CIFAR10(norm_path_to_cifar10)

# Subsample the data for more efficient code execution in this exercise
num_training = 500
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 50
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)


# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=5)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


print('Done.')