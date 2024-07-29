import h5py
import os

import numpy as np
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

from datasets import load_h5, read_h5_dataset
np.random.seed(2000)

pts_, _ = read_h5_dataset('train', 6)

object_density_v1, object_density_v2 = [], []
for pts in tqdm(pts_):

    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(pts)
    distances, indices = nbrs.kneighbors(pts)
    mean_value = np.mean(distances[1:])
    object_density_v1.append(mean_value)
    
object_density_v1 = np.array(object_density_v1)
print(object_density_v1.shape)
print(object_density_v1.min())
print(object_density_v1.max())
print(object_density_v1.mean())

np.save('./data/psi/trainset-object_density.npy', object_density_v1)
