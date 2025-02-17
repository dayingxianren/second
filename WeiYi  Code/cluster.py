import numpy as np
from sklearn.cluster import KMeans
from data import Data

class Clusters:
    def __init__(self, k):
        self.data = Data()
        self.N_packages = self.data.N_packages
        self.Len = self.data.Len
        self.car_start = self.data.car_start
        self.N_special_packages = self.data.N_special_packages
        self.k = k

    def generate_clusters(self, packages):
        # 聚类以确定无人车+无人机方案的中继点
        kmeans_drone = KMeans(n_clusters=self.k)
        kmeans_drone.fit(packages)
        relay_points_drone = kmeans_drone.cluster_centers_
        relay_points_drone = np.round(relay_points_drone)
        relay_points_drone = np.vstack([self.car_start, relay_points_drone])
        return kmeans_drone, relay_points_drone

    def pick_special_packages(self, kmeans_drone, packages, relay_points_drone):
        # 随机选择特殊点
        special_point_indices = np.random.choice(len(packages), size=self.N_special_packages, replace=False)
        special_points = packages[special_point_indices]
        # 确定特殊点所属的聚类
        special_clusters = kmeans_drone.predict(special_points)
        # 初始化一部字典来统计每个聚类中特殊点的个数
        cluster_special_counts = {}
        # 记录特殊点的信息
        special_packages = []
        for i, index in enumerate(special_point_indices):
            cluster_index = special_clusters[i]
            cluster_center = relay_points_drone[cluster_index]
            # 记录特殊点的坐标、所属聚类的索引、以及它们在原始数据中的索引
            special_packages.append({
                'point': special_points[i],
                'cluster_index': cluster_index,
                'cluster_center': cluster_center,
                'original_index': index
            })
            # 更新聚类中特殊点的计数
            if cluster_index in cluster_special_counts:
                cluster_special_counts[cluster_index] += 1
            else:
                cluster_special_counts[cluster_index] = 1

        return special_packages, cluster_special_counts