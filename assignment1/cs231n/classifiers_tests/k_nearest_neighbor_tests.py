import numpy as np
import sys
import os

path_to_this_file = os.path.dirname(os.path.abspath(__file__))
path_to_cs231n = path_to_this_file + "\\..\\.."
path_to_datasets = path_to_cs231n + "\\cs231n"
path_to_cifar10 =  path_to_datasets + "\\datasets\\cifar-10-batches-py"

sys.path.append(path_to_cs231n)
sys.path.append(path_to_datasets)

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
X_train, y_train, X_test, y_test = load_CIFAR10(path_to_cifar10)

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






print("X_train[1]")
print(X_train[1])


print("X_train.size")
print(X_train.size)



print('Done.')