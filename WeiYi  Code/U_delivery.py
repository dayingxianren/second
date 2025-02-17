import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
from sklearn.cluster import KMeans
import os
os.environ['OMP_NUM_THREADS'] = '1'
# 设置参数
np.random.seed(42)  # 固定随机种子
car_start = np.array([0, 0])  # 无人车起点
car_speed = 1                 # 无人车速度：每单位时间1个网格单位
drone_speed = 2               # 无人机速度：每单位时间2个网格单位

N_packages = 100              # 包裹总数
N_special_packages = 10       # 特殊包裹总数

Len = 100                     # 区域的边长
N_drones = 5                  # 无人机数量

max_drone_range = 100         # 无人机的最大飞行航程（曼哈顿距离）

C1 = 20                       # 无人车的能耗参数
C2 = 5                        # 无人机飞行能耗参数
C3 = 10                       # 无人机起降能耗参数

SL = 2                        # 无人机起降对接需要时间
PE = 2000                     # 包裹未送达的惩罚能耗
PT = 200                      # 包裹未送达的惩罚时间
PW = 2000                     # 包裹未送达的惩罚等待时间
SP = 2                        # 特殊点等待时间的权重

min_K = 2                     # 聚类最小值
max_K = 30                    # 聚类最大值

step = 2                      # 聚类数值跨度
mode = 3                      # 价值函数的类型（1：能耗；2：时间；3：等待时间（包含特殊点））

# 定义计算曼哈顿路径长度函数
def calculate_manhattan_path_length(path):
    return sum(cityblock(path[i], path[i+1]) for i in range(len(path)-1))

# 最近邻算法求解TSP，并计算无人车路径总长度
def nearest_neighbor_tsp(points):
    N = len(points)
    visited = [False] * N
    path = [0]
    visited[0] = True
    current = 0
    distances = []  # 存储每两点之间的欧氏距离
    distances_to_clusters = [0]  # 存储起始点到每个聚类点的累积路程
    for _ in range(1, N):
        next_point = np.argmin([cityblock(points[current], points[j]) if not visited[j] else np.inf for j in range(N)])
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

# 无人机路径计算函数，考虑单次配送一个包裹的限制，并增加航程约束
def drone_delivery_paths(relay_point, package_points, special_packages=None):
    paths = [[] for _ in range(N_drones)]
    undelivered_packages = []
    drone_distances = [0] * N_drones     # 记录每个无人机的飞行距离
    drone_round_trips = [0] * N_drones   # 记录每个无人机的往返次数

    special_indices = [point['point'] for point in special_packages]  # 获取特殊点的索引列表
    # 首先处理特殊包裹
    for package in package_points:
        is_special_package = any(np.array_equal(package, cords) for cords in special_indices)
        if is_special_package:
            path_to_package = [relay_point, package, relay_point]
            path_length = calculate_manhattan_path_length(path_to_package)
            if path_length <= max_drone_range:
                min_distance_index = np.argmin(drone_distances)
                if drone_distances[min_distance_index] + path_length * 2 <= max_drone_range:
                    drone_distances[min_distance_index] += path_length * 2
                    drone_round_trips[min_distance_index] += 1
                    paths[min_distance_index].append(path_to_package)
                else:
                    undelivered_packages.append(package)
            else:
                undelivered_packages.append(package)

    # 然后处理其他包裹
    for package in package_points:
        is_special_package = any(np.array_equal(package, cords) for cords in special_indices)
        if not is_special_package:
            path_to_package = [relay_point, package, relay_point]
            path_length = calculate_manhattan_path_length(path_to_package)
            if path_length <= max_drone_range:
                min_distance_index = np.argmin(drone_distances)
                if drone_distances[min_distance_index] + path_length * 2 <= max_drone_range:
                    drone_distances[min_distance_index] += path_length * 2
                    drone_round_trips[min_distance_index] += 1
                    paths[min_distance_index].append(path_to_package)
                else:
                    undelivered_packages.append(package)
            else:
                undelivered_packages.append(package)

    # 创建带有中继点标记的无人机已飞距离字典
    marked_drone_distances = {'relay_point': relay_point, 'drones': []}
    for i, distance in enumerate(drone_distances):
        marked_drone_distances['drones'].append({
            'drone_id': f'drone_{i}',
            'drone_distance': distance,
            'round_trips': drone_round_trips[i]})
    return paths, marked_drone_distances, undelivered_packages

def assign_packages_to_drones(relay_points_drone, packages, distances_to_clusters, kmeans_drone, cluster_special_counts=None, special_packages=None):
    drone_paths = []
    drone_distances = []
    all_undelivered_packages = []
    total_distances_to_clusters = 0
    for i in range(1, len(relay_points_drone)):
        relay_point = relay_points_drone[i]
        cluster_indices = np.where(kmeans_drone.labels_ == i - 1)[0]
        package_points = packages[cluster_indices]
        num_special_packages_in_cluster = cluster_special_counts.get(i - 1, 0)  # 获取该聚类中特殊点的数量
        if len(package_points) > 0:
            drone_paths_for_relay, drone_distances_for_relay, undelivered_packages = drone_delivery_paths(relay_point, package_points, special_packages)
            drone_paths.append(drone_paths_for_relay)
            drone_distances.append(drone_distances_for_relay)
            all_undelivered_packages.extend(undelivered_packages)
            # 增大特殊点对无人车距离的权重，也就是说，特殊点对距离的影响更大
            total_distances_to_clusters += distances_to_clusters[i] * (len(package_points) + SP * num_special_packages_in_cluster)
    return drone_paths, drone_distances, all_undelivered_packages, total_distances_to_clusters

def calculate_total_energy(total_car_distance, drone_paths):
    # 计算无人机的总耗能
    drone_length = 0
    path_count = 0
    for drone_path_group in drone_paths:
        for path_group in drone_path_group:
            for path in path_group:
                drone_path_length = calculate_manhattan_path_length(path)
                drone_length += drone_path_length
                path_count += 1
    # 计算总能耗(无人车总耗能+无人机飞行耗能+无人机起降耗能)
    # 添加系数进行修正
    total_energy = C1 * total_car_distance + C2 * drone_length + C3 * path_count
    return total_energy

def calculate_total_time(total_car_distance, drone_distances):
    # 计算无人车的总时间
    car_time = total_car_distance / car_speed
    # 计算无人机的总时间
    drone_time = 0
    for distances in drone_distances:
        max_drone_distance = max([drone['drone_distance'] + drone['round_trips'] * drone_speed * SL
                                  for drone in distances['drones']])   # 往返次数*对接时间=对接耗费总时间
        drone_time += max_drone_distance / drone_speed
    # 计算总时间(考虑发射接收时间，真·总时间)
    total_time = car_time + drone_time
    return total_time

def calculate_total_waiting_time(total_distances_to_clusters, drone_paths, special_packages=None):
    drone_time = 0    # 计算所有乘客的等待时间
    special_indices = [point['point'] for point in special_packages]  # 获取特殊点的索引列表
    # 严格意义上来说，虽然优先配送特殊包裹，但同时有多个特殊包裹时也会有积压，姑且不考虑
    for drone_path_group in drone_paths:
        for path_group in drone_path_group:
            for path in path_group:
                package_cords = path[1]  # 包裹的坐标
                # 检查包裹坐标是否在特殊点坐标列表中
                is_special_path = any(np.array_equal(package_cords, cords) for cords in special_indices)
                # 应用特殊点权重
                weight = (SP + 1) if is_special_path else 1
                drone_path_length = calculate_manhattan_path_length(path)
                drone_time += (drone_path_length * weight) / (drone_speed * 2)  # 对于实际的等待时间，只有去程没有返程，因此要除以2
    car_time = total_distances_to_clusters / car_speed
    total_time = car_time + drone_time
    return total_time


def plot_path(points, path, color, label):
    path_points = points[path]
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        plt.plot([x1, x2], [y1, y2], color=color, label=label if i == 0 else "")
    plt.scatter(path_points[:, 0], path_points[:, 1], c=color)


def visualize_paths(car_path, drone_paths, relay_points_drone, packages, special_packages=None,
                    all_undelivered_packages=None, package_delivery_order=None):
    plt.figure(figsize=(10, 10))
    plt.title('Delivery Routes')
    # 绘制无人车路径
    plot_path(relay_points_drone, car_path, 'blue', 'Car Path')

    # 绘制无人机路径
    for i, drone_path_group in enumerate(drone_paths):
        for j, path_group in enumerate(drone_path_group):
            for path in path_group:
                drone_path = np.array(path)
                plt.plot(drone_path[:, 0], drone_path[:, 1], 'cyan', linestyle='--',
                         label=f'Drone {i + 1} Path' if j == 0 else "")

    # 绘制所有包裹点
    for i, package in enumerate(packages):
        plt.scatter(package[0], package[1], c='red', label='Packages' if i == 0 else "")
        if package_delivery_order is not None:
            plt.text(package[0], package[1], str(i + 1), fontsize=9, ha='right')

    # 绘制特殊包裹点
    if special_packages:
        for package in special_packages:
            plt.scatter(package['point'][0], package['point'][1], c='yellow', marker='*',
                        label='Special Packages' if i == 0 else "")

    # 绘制未送达包裹点
    if all_undelivered_packages:
        for package in all_undelivered_packages:
            plt.scatter(package[0], package[1], c='black', marker='x', label='Undelivered Packages' if i == 0 else "")

    plt.legend()
    plt.grid(True)
    plt.show()


def pick_special_packages(kmeans_drone, packages, relay_points_drone):
    special_point_indices = np.random.choice(len(packages), size=N_special_packages, replace=False)
    special_points = packages[special_point_indices]
    special_clusters = kmeans_drone.predict(special_points)
    cluster_special_counts = {}
    special_packages = []
    for i, index in enumerate(special_point_indices):
        cluster_index = special_clusters[i]
        cluster_center = relay_points_drone[cluster_index]
        special_packages.append({
            'point': special_points[i],
            'cluster_index': cluster_index,
            'cluster_center': cluster_center,
            'original_index': index
        })
        if cluster_index in cluster_special_counts:
            cluster_special_counts[cluster_index] += 1
        else:
            cluster_special_counts[cluster_index] = 1
    return special_packages, cluster_special_counts


if __name__ == '__main__':
    # 生成包裹位置（含小数），固定随机种子以确保位置不变
    packages = np.random.rand(N_packages, 2) * Len

    # 假设 packages 是一个包含包裹坐标的列表
    for i, package in enumerate(packages, 1):
        print(f"package_{i}: {tuple(package)}")

    # 添加一个变量来存储最低能耗和对应的K值
    min_energy = float('inf')
    min_time = float('inf')
    min_waiting_time = float('inf')
    energy = float('inf')
    time = float('inf')
    waiting_time = float('inf')
    best_k = 0
    best_car_path = None
    best_distances_to_clusters = None
    best_drone_paths = None
    best_relay_points_drone = None
    best_packages = None
    best_undelivered_packages = None
    best_special_packages = None
    best_package_delivery_order = None

    # 循环遍历不同的K值
    for k in range(min_K, max_K, step):  # 可以根据需要调整K值的范围和步长
        # 聚类以确定无人车+无人机方案的中继点
        kmeans_drone = KMeans(n_clusters=k, random_state=42)
        kmeans_drone.fit(packages)
        relay_points_drone = kmeans_drone.cluster_centers_
        relay_points_drone = np.vstack((car_start, relay_points_drone))  # 添加无人车起点

        if mode == 3:
            special_packages, cluster_special_counts = pick_special_packages(kmeans_drone, packages, relay_points_drone)

        # 计算无人车路径（TSP）
        car_path, total_car_distance, distances_to_clusters = nearest_neighbor_tsp(relay_points_drone)

        # 计算无人机路径并记录无法配送的包裹
        drone_paths, drone_distances, all_undelivered_packages, total_distances_to_clusters = assign_packages_to_drones(
            relay_points_drone, packages,
            distances_to_clusters, kmeans_drone,
            cluster_special_counts, special_packages)

        # 记录每个包裹的送达顺序
        package_delivery_order = []
        for drone_path_group in drone_paths:
            for path_group in drone_path_group:
                for path in path_group:
                    for package in path[1:-1]:  # 排除起始和结束的中继点
                        package_delivery_order.append(package)

        # 计算总能耗
        if mode == 1:
            basic_energy = calculate_total_energy(total_car_distance, drone_paths)
            penalty_energy = PE * len(all_undelivered_packages)  # 计算包裹未送达的惩罚
            energy = basic_energy + penalty_energy
        elif mode == 2:
            basic_time = calculate_total_time(total_car_distance, drone_distances)
            penalty_time = PT * len(all_undelivered_packages)
            time = basic_time + penalty_time
        elif mode == 3:
            basic_waiting_time = calculate_total_waiting_time(total_distances_to_clusters, drone_paths,
                                                              special_packages)
            penalty_waiting_time = PW * len(all_undelivered_packages)
            waiting_time = basic_waiting_time + penalty_waiting_time

        # 更新最低时间能耗和对应的K值
        if (mode == 1 and energy < min_energy) or (mode == 2 and time < min_time) or (
                mode == 3 and waiting_time < min_waiting_time):
            min_energy = energy
            min_time = time
            min_waiting_time = waiting_time
            best_k = k
            best_car_path = car_path
            best_distances_to_clusters = distances_to_clusters
            best_drone_paths = drone_paths
            best_relay_points_drone = relay_points_drone
            best_undelivered_packages = all_undelivered_packages
            best_special_packages = special_packages
            best_package_delivery_order = package_delivery_order

    print(f"Best K: {best_k}")
    print(f"Minimum Energy: {min_energy}")
    print(f"Minimum Time: {min_time}")
    print(f"Minimum Waiting Time: {min_waiting_time}")
    # 使用能耗(时间)最低的K值对应的路径进行绘图
    visualize_paths(best_car_path, best_drone_paths, best_relay_points_drone, packages, best_special_packages,
                    best_undelivered_packages, best_package_delivery_order)