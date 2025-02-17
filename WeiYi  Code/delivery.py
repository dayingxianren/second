import numpy as np
from scipy.spatial.distance import cityblock
from data import Data

class Delivery:
    def __init__(self):
        self.data = Data()

    # 定义计算曼哈顿路径长度函数
    def calculate_manhattan_path_length(self, path):
        return sum(cityblock(path[i], path[i + 1]) for i in range(len(path) - 1))

    def nearest_neighbor_tsp(self, points):
        N = len(points)
        visited = [False] * N
        path = [0]
        visited[0] = True
        current = 0
        distances = []  # 存储每两点之间的欧氏距离
        distances_to_clusters = [0]  # 存储起始点到每个聚类点的累积路程
        for _ in range(1, N):
            next_point = np.argmin(
                [cityblock(points[current], points[j]) if not visited[j] else np.inf for j in range(N)])
            distance_to_next_point = np.linalg.norm(points[current] - points[next_point])  # 当前点到下一个点的距离
            distances.append(distance_to_next_point)  # 存储当前点到下一个点的距离
            distances_to_clusters.append(distances_to_clusters[-1] + distance_to_next_point)  # 更新累积路程
            path.append(next_point)
            visited[next_point] = True
            current = next_point
        # 计算返回起点的距离
        distances.append(np.linalg.norm(points[current] - points[0]))
        path.append(0)  # 返回起点
        total_car_distance = sum(distances)  # 计算总距离

        return path, total_car_distance, distances_to_clusters

    def assign_packages_to_drones(self, relay_points_drone, packages, distances_to_clusters, kmeans_drone,
                                  cluster_special_counts=None, special_packages=None):
        drone_paths = []
        drone_distances = []
        all_undelivered_packages = []
        total_distances_to_clusters = 0
        for i in range(1, len(relay_points_drone)):
            relay_point = relay_points_drone[i]
            cluster_indices = np.where(kmeans_drone.labels_ == i - 1)[0]
            package_points = packages[cluster_indices]
            num_special_packages_in_cluster = (cluster_special_counts.get(i - 1, 0) if self.data.mode == 3 else 0) # 获取该聚类中特殊点的数量
            if len(package_points) > 0:
                drone_paths_for_relay, drone_distances_for_relay, undelivered_packages = self.drone_delivery_paths(
                    relay_point, package_points, special_packages)
                drone_paths.append(drone_paths_for_relay)
                drone_distances.append(drone_distances_for_relay)
                all_undelivered_packages.extend(undelivered_packages)

                # 增大特殊点对无人车距离的权重，也就是说，特殊点对距离的影响更大
                total_distances_to_clusters += (distances_to_clusters[i] *
                                                (self.data.SW * len(package_points) + self.data.SP * num_special_packages_in_cluster))
        return drone_paths, drone_distances, all_undelivered_packages, total_distances_to_clusters


    # 无人机路径计算函数，考虑单次配送一个包裹的限制，并增加航程约束
    def drone_delivery_paths(self, relay_point, package_points, special_packages):
        paths = [[] for _ in range(self.data.N_drones)]
        undelivered_packages = []
        drone_distances = [0] * self.data.N_drones  # 记录每个无人机的飞行距离
        drone_round_trips = [0] * self.data.N_drones  # 记录每个无人机的往返次数

        special_indices = ([point['point'] for point in special_packages] if self.data.mode == 3 else None) # 获取特殊点的索引列表
        # 首先处理特殊包裹
        for package in package_points:
            is_special_package = (any(np.array_equal(package, cords) for cords in special_indices)if self.data.mode == 3 else False)
            if is_special_package:
                paths, undelivered_packages, drone_distances, drone_round_trips \
                    = self.get_drone_paths(relay_point, package, drone_distances,
                                           drone_round_trips, undelivered_packages, paths)
        # 然后处理其他包裹
        for package in package_points:
            is_special_package = (any(np.array_equal(package, cords) for cords in special_indices)if self.data.mode == 3 else False)
            if not is_special_package:
                paths, undelivered_packages, drone_distances, drone_round_trips \
                    = self.get_drone_paths(relay_point, package, drone_distances,
                                           drone_round_trips,undelivered_packages, paths)

        # 创建带有中继点标记的无人机已飞距离字典
        marked_drone_distances = {'relay_point': relay_point, 'drones': []}
        for i, distance in enumerate(drone_distances):
            marked_drone_distances['drones'].append({
                'drone_id': f'drone_{i}',
                'drone_distance': distance,
                'round_trips': drone_round_trips[i]})

        return paths, marked_drone_distances, undelivered_packages

    def get_drone_paths(self, relay_point, package, drone_distances, drone_round_trips, undelivered_packages, paths):
        path_to_package = [relay_point, package, relay_point]
        path_length = self.calculate_manhattan_path_length(path_to_package)
        if path_length <= self.data.max_drone_range:
            min_distance_index = np.argmin(drone_distances)
            if drone_distances[min_distance_index] + path_length * 2 <= self.data.max_drone_range:
                drone_distances[min_distance_index] += path_length * 2
                drone_round_trips[min_distance_index] += 1
                paths[min_distance_index].append(path_to_package)
            else:
                undelivered_packages.append(package)
        else:
            undelivered_packages.append(package)

        return paths, undelivered_packages, drone_distances, drone_round_trips