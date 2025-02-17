import numpy as np
from data import Data
from cluster import Clusters
from delivery import Delivery

class Values:
    def __init__(self, k):
        self.data = Data()
        self.clusters = Clusters(k)
        self.delivery = Delivery()

    def calculate_total_energy(self, total_car_distance, drone_paths):
        # 计算无人机的总耗能
        drone_length = 0
        path_count = 0
        for drone_path_group in drone_paths:
            for path_group in drone_path_group:
                for path in path_group:
                    drone_path_length = self.delivery.calculate_manhattan_path_length(path)
                    drone_length += drone_path_length
                    path_count += 1
        # 添加系数进行修正 # 计算总能耗(无人车总耗能+无人机飞行耗能+无人机起降耗能)
        total_energy = self.data.C1 * total_car_distance + self.data.C2 * drone_length + self.data.C3 * path_count
        return total_energy

    def calculate_total_time(self, total_car_distance, drone_distances):
        # 计算无人车的总时间
        car_time = total_car_distance / self.data.car_speed
        # 计算无人机的总时间
        drone_time = 0
        for distances in drone_distances:
            max_drone_distance = max([drone['drone_distance'] + drone['round_trips'] * self.data.drone_speed * self.data.SL
                                      for drone in distances['drones']])  # 往返次数*对接时间=对接耗费总时间
            drone_time += max_drone_distance / self.data.drone_speed
        # 计算总时间(考虑发射接收时间，真·总时间)
        total_time = car_time + drone_time
        return total_time

    def calculate_total_waiting_time(self, total_distances_to_clusters, drone_paths, special_packages=None):
        drone_time = 0  # 计算所有乘客的等待时间
        special_indices = [point['point'] for point in special_packages]  # 获取特殊点的索引列表
        # 严格意义上来说，虽然优先配送特殊包裹，但同时有多个特殊包裹时也会有积压，姑且不考虑
        for drone_path_group in drone_paths:
            for path_group in drone_path_group:
                for path in path_group:
                    package_cords = path[1]  # 包裹的坐标
                    # 检查包裹坐标是否在特殊点坐标列表中
                    is_special_path = any(np.array_equal(package_cords, cords) for cords in special_indices)
                    # 应用特殊点权重
                    weight = (self.data.SP + self.data.SW) if is_special_path else self.data.SW
                    drone_path_length = self.delivery.calculate_manhattan_path_length(path)
                    drone_time += (drone_path_length * weight) / (self.data.drone_speed * 2)  # 对于实际的等待时间，只有去程没有返程，因此要除以2
        car_time = total_distances_to_clusters / self.data.car_speed
        total_waiting_time = car_time + drone_time
        # 注意，这里的等待时间不是真实的等待时间，而是加权后的结果
        return total_waiting_time

