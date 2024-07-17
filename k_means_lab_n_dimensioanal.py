'''
import numpy as np
import plotly.express as px

def init_centroids(num_clusters: int, data: np.ndarray) -> np.ndarray:
   return data[np.random.choice(data.shape[0], num_clusters, replace=False)]

def centr_dist(data:np.ndarray, centroid:np.ndarray)->np.ndarray:
   return np.linalg.norm(data - centroid, axis=1)

def k_means(data:np.ndarray, num_clasters:int, n_iter:int=20)->tuple:
   'Returns labels, centroids, history'
   centroids = init_centroids(num_clasters, data)

   centr_history = []
   for _ in range(n_iter):
      labels = np.argmin(np.array([centr_dist(data, centroids[i]) for i in range(num_clasters)]).T, axis=1)
      centr_history.append(centroids)
      centroids = np.array([coords[labels==i].mean(axis=0) for i in range(num_clasters)])

   labels = np.argmin(np.array([centr_dist(data, centroids[i]) for i in range(num_clasters)]).T, axis=1)
   history = np.array(centr_history)
   return labels, centroids, history

N_CLASTERS = 3
N_ITER = 25

coords = np.concatenate([np.random.randn(100, 4)+np.array([1, 2, -1, 0]),
                         np.random.randn(100, 4)+np.array([-5, 4, 2, 1]),
                         np.random.randn(150, 4)+np.array([3, -3, 0, 2]),
                         np.random.randn(150, 4)+np.array([7, 1, -10, 3])], axis=0)

labels, centroids, history = k_means(coords, N_CLASTERS, N_ITER)

coords.shape
print('labels:', labels)
print('centroids:', centroids)

fig = px.scatter_3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 3],
              color=labels)
fig.show()
'''
import numpy as np
import matplotlib.pyplot as plt

def init_centroids(num_clusters: int, data: np.ndarray) -> np.ndarray:
    return data[np.random.choice(data.shape[0], num_clusters, replace=False)]

def centr_dist(data: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.linalg.norm(data - centroid, axis=1)

N_CLUSTERS = 4
N_ITER = 100
TOL = 1e-4

coords = np.concatenate([np.random.randn(100, 4)+np.array([1, 2, -1, 0]),
                         np.random.randn(100, 4)+np.array([-5, 4, 2, 1]),
                         np.random.randn(150, 4)+np.array([3, -3, 0, 2]),
                         np.random.randn(150, 4)+np.array([7, 1, -10, 3])], axis=0)

centroids = init_centroids(N_CLUSTERS, coords)
centr_history = []
print('centroids:', centroids)
for _ in range(N_ITER):
    distances = np.array([centr_dist(coords, centroids[i]) for i in range(N_CLUSTERS)]).T
    labels = np.argmin(distances, axis=1)
    
    centr_history.append(centroids.copy())
    
    new_centroids = np.array([coords[labels == i].mean(axis=0) for i in range(N_CLUSTERS)])
    
    if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < TOL):
        break
    
    centroids = new_centroids

labels = np.argmin(np.array([centr_dist(coords, centroids[i]) for i in range(N_CLUSTERS)]).T, axis=1)
print('labels:', labels)
print('centr_history:', centr_history)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'green', 'purple', 'orange']
for i in range(N_CLUSTERS):
    points = coords[labels == i]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, c=colors[i], label=f'Cluster {i+1}')
    
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=300, c='red', marker='X', label='Centroids')


ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('K-means Clustering in 3D')
ax.legend()
plt.show()