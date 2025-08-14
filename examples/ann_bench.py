"""
Copyright (c) 2021 The authors of SG Tree All rights reserved.

Initially modified from CoverTree
https://github.com/manzilzaheer/CoverTree
Copyright (c) 2017 Manzil Zaheer All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import numpy as np

from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from graphgrove.covertree import NNS_L2 as CoverTree_NNS_L2
import h5py

gt = time.time

np.random.seed(123)
cores = 16

hdf5_filename = "../data/fashion-mnist-784-euclidean.hdf5"

print('======== Building Dataset ==========')
hdf5_file = h5py.File(hdf5_filename, "r")
train_set = hdf5_file["train"]

test_set = hdf5_file["test"]
neighbors = hdf5_file["neighbors"]
distances = hdf5_file["distances"]

x = np.require(train_set, requirements=['A', 'C', 'O', 'W'])
y = np.require(test_set, requirements=['A', 'C', 'O', 'W'])

print(x.dtype)
print('======== SG Tree ==========')
t = gt()
ct = SGTree_NNS_L2.from_matrix(x, use_multi_core=cores)
b_t = gt() - t
print(ct.stats())
#ct.display()
print("Building time:", b_t, "seconds")


"""
print('Test k-Nearest Neighbours - Exact (k=3): ')
t = gt()
idx1, d1 = ct.kNearestNeighbours(y, 1, use_multi_core=cores)
b_t = gt() - t
print("Query time - Exact:", b_t, "seconds")
print(np.mean(idx1[:, 0] == neighbors[:, 0]))

print('======== Cover Tree ==========')
t = gt()
ct = CoverTree_NNS_L2.from_matrix(x, use_multi_core=cores)
b_t = gt() - t
#ct.display()
print("Building time:", b_t, "seconds")
"""
