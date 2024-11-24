import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# 读取CSV文件
data = pd.read_csv('datasets/unlabeled/left_lzp_data.csv', sep=' ', header=None)

# 提取加速度和压力数据
features = data.iloc[:, 1:].values  # 忽略第一个时间戳列

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

# 聚类结果
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用前三个特征进行3D可视化
scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='viridis', label=labels)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=300, marker='X', label='Centroids')

ax.set_xlabel('Acceleration X')
ax.set_ylabel('Acceleration Y')
ax.set_zlabel('Acceleration Z')
ax.set_title('K-means Clustering of Insole Data')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()