import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

coords = np.concatenate([np.random.randn(100, 2)+np.array([1, 2]),
                         np.random.randn(100, 2)+np.array([-5, 4]),
                         np.random.randn(150, 2)+np.array([3, -5])], axis=0)

sns.scatterplot(x=coords[:, 0], y=coords[:, 1])

"""## Algorithm"""

def init_centriods(num_clasters:int, data:np.ndarray)->np.ndarray:
  return np.random.randint(data.min(), data.max(), (num_clasters, data.shape[1]))*1

def centr_dist(data:np.ndarray, centroid:np.ndarray)->np.ndarray:
  return np.linalg.norm(data - centroid, axis=1)

N_CLASTERS = 3
N_ITER = 10

centroids = init_centriods(3, coords)
centroids

centr_history = []
for _ in range(N_ITER):
  labels = np.argmin(np.array([centr_dist(coords, centroids[i]) for i in range(N_CLASTERS)]).T, axis=1)
  centr_history.append(centroids)
  centroids = np.array([coords[labels==i].mean(axis=0) for i in range(N_CLASTERS)])

labels = np.argmin(np.array([centr_dist(coords, centroids[i]) for i in range(N_CLASTERS)]).T, axis=1)

centr_history = np.array(centr_history)

centr_history.shape

centr_history[:, 0, :]

sns.lineplot(x = centr_history[:, 0, 0], y = centr_history[:, 0, 1], marker='o')
sns.lineplot(x = centr_history[:, 1, 0], y = centr_history[:, 1, 1], marker='o')
sns.lineplot(x = centr_history[:, 2, 0], y = centr_history[:, 2, 1], marker='o')

sns.scatterplot(x=coords[:, 0], y=coords[:, 1],)
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red',)

sns.lineplot(x = centr_history[:, 0, 0], y = centr_history[:, 0, 1], marker='o', color='black')
sns.lineplot(x = centr_history[:, 1, 0], y = centr_history[:, 1, 1], marker='o', color='black')
sns.lineplot(x = centr_history[:, 2, 0], y = centr_history[:, 2, 1], marker='o', color='black')


plt.show()

