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
import pickle
import numpy as np

from functools import partial

from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from graphgrove.covertree import NNS_L2 as CoverTree_NNS_L2

gt = time.time

np.random.seed(123)
cores = 16


data_dir = "/drive_sdc/ssarup/faiss_data/"
hyper_pass_path = f"{data_dir}data/passage_hyperbolic_embeddings_128dims_regularized_variance.npy"
hyper_query_path = f"{data_dir}data/query_hyperbolic_embeddings_128dims_regularized_variance.npy"

with open(f"{data_dir}data/train_gold_dict.pkl", "rb") as f:
    qidx2gold_pidx = pickle.load(f)

data_dir = "/drive_sdc/deviyer/faiss_data/"
hyper_query = np.load(hyper_query_path)
hyper_pass = np.load(hyper_pass_path)

q = np.array(list(qidx2gold_pidx.keys()))

print(np.min(q), np.max(q), q.shape)
print("[setup] Generating answers...")
max_p = max([len(v) for v in qidx2gold_pidx.values()])
a = np.full((hyper_query.shape[0], max_p), -2)
for k, v in qidx2gold_pidx.items():
    a[k, : len(v)] = np.array(list(v))
print("[setup] Done!")

d = hyper_pass.shape[-1]
K = 1001

mini_D = 1000
mini_Q = 50
mini_pass = hyper_pass[:mini_D]
mini_query = hyper_query[:mini_Q]
mini_q = q[:mini_Q]
mini_a = a[mini_q]
#mini_pass = hyper_pass[mini_a[:, 0]]
mini_pass = np.concat((mini_pass, hyper_pass[mini_a[:, 0]]))
mini_a_test = np.full((hyper_query.shape[0], max_p), -2)
mini_a_test[mini_q, 0] = np.arange(mini_Q) + mini_D
#mini_a_test[mini_q, 0] = np.arange(mini_Q)

def succ_at(K, I, a, queries):
    return np.mean((I[:, :K, None] == a[queries, None, :]).any(axis=(-1, -2)))


def test_index(ct, pass_data, query_data, queries, a=a):
    print("Beginning Search")
    t = gt()
    I, d1 = ct.kNearestNeighbours(query_data[queries], 10, use_multi_core=0)
    print(I)
    b_t = gt() - t
    print("Done searching in %.2f seconds" % (b_t))

    t = gt()
    r1 = succ_at(1, I, a, queries)
    r10 = succ_at(10, I, a, queries)
    r100 = succ_at(100, I, a, queries)
    r1000 = succ_at(1000, I, a, queries)
    b_t = gt() - t

    print("Done calculating succ in %.2f seconds" % (b_t))
    print(f"Success@1: {r1:.4f}")
    print(f"Success@10: {r10:.4f}")
    print(f"Success@100: {r100:.4f}")
    print(f"Success@1000: {r1000:.4f}")


mini_pass = np.require(mini_pass.astype(np.float32), requirements=['A', 'C', 'O', 'W'])

hyper_query = np.require(hyper_query.astype(np.float32), requirements=['A', 'C', 'O', 'W'])
hyper_pass = np.require(hyper_pass.astype(np.float32), requirements=['A', 'C', 'O', 'W'])

def dist_mat(vec):
    norms = 1 - np.linalg.norm(vec, axis=-1)
    norm2 = norms[None, :]*norms[:, None]
    dists = np.linalg.norm(vec[None, :, :] - vec[:, None, :], axis=-1)/norm2
    return dists + np.eye(dists.shape[0])
#dists = dist_mat(hyper_pass)
#print("DUPLICATES")
"""
pass_norms = 1 - np.linalg.norm(mini_pass, axis=-1)
query_data = hyper_query[mini_q]
query_norms = np.linalg.norm(query_data, axis=-1)
hyper_dist = np.linalg.norm(query_data[:, None, :] - mini_pass[None, :, :], axis=-1)/(query_norms[:, None]*pass_norms[None, :])
print(np.mean(np.argmin(hyper_dist, axis=-1) == np.arange(mini_Q)))
"""
"""
print("Computing Norms...")
pass_norms = 1 -  np.linalg.norm(hyper_pass, axis=-1)
norm_idx = np.argsort(pass_norms)
#print(pass_norms[norm_idx[-100:]])
"""

#sorted_hyper_pass = hyper_pass[norm_idx[:50001]]
print('======== SG Tree ==========')
t = gt()
# Note that this rn uses the mean of the points in euclidean space, which may not be the "mean in hyperbolic space" (need to weight by the norms, probably)
base = 1.3
scale = 19
ct = SGTree_NNS_L2.from_matrix(hyper_pass[:100000], use_multi_core=cores, base=base, scale=scale)
b_t = gt() - t
print("Building time:", b_t, "seconds")
with open(f'ct_{base:0.2f}_{scale}.bin', 'wb+') as f:
    f.write(ct.serialize())
"""
with open(f'ct_{base:0.2f}_{scale}.bin', 'rb') as f:
    ct = SGTree_NNS_L2.from_string(f.read())
"""
q_data = np.array([hyper_query[q[0]]])
res = ct.NearestNeighbour(q_data, use_multi_core=0)
print(res)
res = ct.kNearestNeighbours(q_data, 10, use_multi_core=0)
print(res)
test_index(ct, hyper_pass, hyper_query, q[-1024:], a=a)
"""
test_index(ct, hyper_pass, hyper_query, q[:1024], a=a)
"""

