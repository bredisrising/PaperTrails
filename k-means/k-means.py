import numpy as np
import matplotlib.pyplot as plt
import time
import random


K = 3
SPREAD = .2

num_clusters = random.randint(5, 10)
points_per_cluster = 100

x_spread = np.random.rand(num_clusters, 1)
y_spread = x_spread

x = np.random.randn(num_clusters, points_per_cluster) * x_spread * SPREAD + np.random.rand(num_clusters, 1)
y = np.random.randn(num_clusters, points_per_cluster) * y_spread * SPREAD + np.random.rand(num_clusters, 1)


cluster_centers = np.random.rand(K, 2)

xs = x.ravel()
ys = y.ravel()


old_center = cluster_centers.copy()

step = 0
while True:
    step += 1
    print(f"{step}", end="\r")
    cluster_points = [[] for i in range(K)]

    for px, py in zip(xs, ys):
        min_dist = 1000000
        min_index  = -1
        for i in range(K): 
            point = cluster_centers[i]

            dist = (point[0] - px) ** 2 + (point[1] - py) ** 2 
                
            if dist < min_dist:
                min_dist = dist
                min_index = i

        cluster_points[min_index].append((px, py))

    for i in range(K):
        avg_x = 0
        avg_y = 0
        for point in cluster_points[i]:
            avg_x += point[0] 
            avg_y += point[1] 
        avg_x /= max(len(cluster_points[i]), 1)
        avg_y /= max(len(cluster_points[i]), 1)
    
        cluster_centers[i][0] = avg_x
        cluster_centers[i][1] = avg_y
    
    delta = np.linalg.norm(cluster_centers - old_center, axis=1)
    if delta.max() < 1e-6:
        break  

    old_center = cluster_centers.copy()
    

plt.scatter(x, y)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red')
plt.show()






