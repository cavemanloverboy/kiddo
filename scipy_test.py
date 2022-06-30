from scipy.spatial import cKDTree as Tree
import numpy as np
from time import time

Nruns = 10


building = 0
querying = 0

Nd = 10**5
Nq = 10**5
for _ in range(Nruns):

    data = np.random.uniform(size=(Nd, 3)).astype(np.float64)
    query = np.random.uniform(size=(Nq, 3)).astype(np.float64)

    start = time()
    tree = Tree(data, compact_nodes=False, balanced_tree=False, leafsize=32)
    building += time() - start

    start = time()
    r, ids = tree.query(query, k=1, workers=1)
    querying += time() - start

avg_query_time = querying / Nruns * 1000
avg_build_time = building / Nruns * 1000
print(f"finished non-pbc with {avg_query_time} millis average querying, {avg_build_time} building")


building = 0
querying = 0
for _ in range(Nruns):

    data = np.random.uniform(size=(Nd, 3)).astype(np.float64)
    query = np.random.uniform(size=(Nq, 3)).astype(np.float64)

    start = time()
    tree = Tree(data, compact_nodes=False, balanced_tree=False, leafsize=32, boxsize=[1.0, 1.0, 1.0])
    building += time() - start

    start = time()
    r, ids = tree.query(query, k=1, workers=1)
    querying += time() - start

avg_query_time = querying / Nruns * 1000
avg_build_time = building / Nruns * 1000
print(f"finished     pbc with {avg_query_time} millis average querying, {avg_build_time} building")

