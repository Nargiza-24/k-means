import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

centroids = np.array([[5, 2], [-5, 7], [0, -6]])
coords = np.concatenate([np.random.randn(100, 2)+np.array([1, 2])+centroids[0],
                         np.random.randn(100, 2)+np.array([-5, 4])+centroids[1],
                         np.random.randn(150, 2)+np.array([3, -5])+centroids[2]], axis=0)
coords[:5]

sns.scatterplot(x = coords[:, 0], y = coords[:, 1], color='green')
sns.scatterplot(x = centroids[:, 0], y = centroids[:, 1], color='red')

def get_distance(matrix:np.ndarray, centroid:np.ndarray)->np.ndarray:
    return np.linalg.norm(matrix - centroid, axis=1)

np.array([get_distance(coords, centroids[0]),get_distance(coords, centroids[1]),get_distance(coords, centroids[2])]).T

dist1 = get_distance(coords, centroids[0])
dist2 = get_distance(coords, centroids[1])
dist3 = get_distance(coords, centroids[2])
print(dist1[:5])
print(dist2[:5])
print(dist3[:5])

labels=np.where(dist1 < dist2, np.where(dist1 < dist3, 0, 2), np.where(dist2 < dist3, 1, 2))

labels.shape

labels=np.argmin(np.array([dist1, dist2, dist3]), axis=0)

sns.scatterplot(x = coords[:, 0], y = coords[:, 1], color='green', hue=labels)
sns.scatterplot(x = centroids[:, 0], y = centroids[:, 1], color='red')

distances = np.sqrt(((coords - centroids[:, np.newaxis])**2).sum(axis=2))
np.argmin(distances, axis=0)

new_centroids = np.array([coords[labels == i].mean(axis=0) for i in range(3)])
new_centroids

N_CLASTERS = 3

[coords[labels == i].mean(axis=0) for i in range(N_CLASTERS)]

sns.scatterplot(x = coords[:, 0], y = coords[:, 1], color='green', hue=labels)
sns.scatterplot(x = new_centroids[:, 0], y = new_centroids[:, 1], color='red')

plt.show()