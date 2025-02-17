from data import Data
from cluster import Clusters
from delivery import Delivery
from value import Values
from draw import Draw
from package import Packages
import matplotlib.pyplot as plt
import numpy as np  # 如果需要处理包裹的位置

if __name__ == '__main__':
    data = Data()
    draw = Draw()
    packages = Packages()

    # 生成包裹位置，并固定包裹位置（你可以选择将其保存到文件中或使用一个固定的种子）
    package_positions = packages.generate_packages()  # 这是原始生成包裹的位置

    # 假设生成的包裹位置为固定的，如果你想固定位置，可以直接硬编码一个固定的数组
    # 示例：假设生成的包裹位置为 numpy 数组
    fixed_package_positions = np.array([[2, 3], [5, 8], [1, 7], [4, 2], [7, 6]])  # 固定的包裹位置

    # 添加一个变量来存储最低能耗和对应的K值
    min_energy = float('inf')
    min_time = float('inf')
    min_waiting_time = float('inf')
    energy = float('inf')
    time = float('inf')
    waiting_time = float('inf')
    best_k = 0
    best_car_path = None
    best_drone_paths = None
    best_relay_points_drone = None
    best_packages = None
    best_undelivered_packages = None
    best_special_packages = None

    # 循环遍历不同的K值
    for k in range(data.min_K, data.max_K, data.step):  # 可以根据需要调整K值的范围和步长
        clusters = Clusters(k)
        delivery = Delivery()
        values = Values(k)

        # 聚类以确定无人车+无人机方案的中继点
        kmeans_drone, relay_points_drone = clusters.generate_clusters(fixed_package_positions)  # 使用固定位置

        if data.mode == 3:
            special_packages, cluster_special_counts = clusters.pick_special_packages(kmeans_drone,
                                                                                      fixed_package_positions,
                                                                                      relay_points_drone)
        else:
            special_packages, cluster_special_counts = None, None

        # 计算无人车路径（TSP）
        car_path, total_car_distance, distances_to_clusters = delivery.nearest_neighbor_tsp(relay_points_drone)
        # 计算无人机路径并记录无法配送的包裹
        drone_paths, drone_distances, all_undelivered_packages, total_distances_to_clusters = delivery.assign_packages_to_drones(
            relay_points_drone, fixed_package_positions,
            distances_to_clusters, kmeans_drone,
            cluster_special_counts, special_packages)
        # 计算总能耗
        if data.mode == 1:
            basic_energy = values.calculate_total_energy(total_car_distance, drone_paths)
            penalty_energy = data.PE * len(all_undelivered_packages)  # 计算包裹未送达的惩罚
            energy = basic_energy + penalty_energy
        elif data.mode == 2:
            basic_time = values.calculate_total_time(total_car_distance, drone_distances)
            penalty_time = data.PT * len(all_undelivered_packages)
            time = basic_time + penalty_time
        elif data.mode == 3:
            basic_waiting_time = values.calculate_total_waiting_time(total_distances_to_clusters, drone_paths,
                                                                     special_packages)
            penalty_waiting_time = data.PW * len(all_undelivered_packages)
            waiting_time = basic_waiting_time + penalty_waiting_time

        # 更新最低时间能耗和对应的K值
        if (data.mode == 1 and energy < min_energy) or (data.mode == 2 and time < min_time) or (
                data.mode == 3 and waiting_time < min_waiting_time):
            min_energy = energy
            min_time = time
            min_waiting_time = waiting_time
            best_k = k
            best_car_path = car_path
            best_drone_paths = drone_paths
            best_relay_points_drone = relay_points_drone
            best_undelivered_packages = all_undelivered_packages
            best_special_packages = special_packages

    print(best_k)
    print(min_energy)
    print(min_time)
    print(min_waiting_time)

    # 使用能耗(时间)最低的K值对应的路径进行绘图
    draw.visualize_paths(best_car_path, best_drone_paths, best_relay_points_drone, fixed_package_positions,
                         best_special_packages, best_undelivered_packages)

    # 绘制包裹位置和标注序号
    plt.figure(figsize=(10, 8))
    # 绘制包裹的点
    package_x, package_y = zip(*fixed_package_positions)
    plt.scatter(package_x, package_y, color='red', label="Packages", zorder=5)

    # 添加序号标签
    for i, (x, y) in enumerate(fixed_package_positions):
        plt.text(x, y, f'{i + 1}', fontsize=12, ha='right', color='black', zorder=10)

    plt.title("Package Delivery Locations with Sequence Numbers")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()
