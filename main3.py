import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
from sklearn.cluster import KMeans
import os
import argparse
import sys

# 设置环境变量以限制OpenMP线程数，避免多线程带来的性能问题
os.environ['OMP_NUM_THREADS'] = '1'

def parse_special_package(s):
    """
    解析特殊包裹的输入格式，期望格式为 'index:weight'，其中weight在1到5之间。

    参数:
        s (str): 输入的特殊包裹字符串，如 '10:3'。

    返回:
        tuple: (index, weight) 的元组。

    异常:
        argparse.ArgumentTypeError: 如果输入格式不正确或权重不在1到5之间。
    """
    try:
        index, weight = s.split(':')
        index = int(index)
        weight = int(weight)
        if not (1 <= weight <= 5):
            raise argparse.ArgumentTypeError(f"Weight must be between 1 and 5. Got {weight}.")
        return (index, weight)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid special package format: '{s}'. Expected format: index:weight")

def set_parameters(args):
    """
    设置并返回所有的全局参数。

    参数:
        args (argparse.Namespace): 从命令行解析得到的参数对象。

    返回:
        dict: 包含所有参数的字典。
    """
    np.random.seed(42)  # 固定随机种子以确保结果可重复
    params = {
        'car_start': np.array([0, 0]),  # 无人车起点坐标
        'car_speed': 1,                  # 无人车速度（单位：网格/时间单位）
        'drone_speed': 2,                # 无人机速度（单位：网格/时间单位）
        'N_packages': 100,               # 包裹总数
        'N_special_packages': len(args.special_packages),  # 特殊包裹数量，根据用户输入
        'Len': 100,                      # 区域边长（假设为正方形区域）
        'N_drones': 5,                   # 无人机数量
        'max_drone_range': 100,          # 无人机的最大飞行航程（曼哈顿距离）
        'C1': 20,                        # 无人车的能耗参数
        'C2': 5,                         # 无人机飞行能耗参数
        'C3': 10,                        # 无人机起降能耗参数
        'SL': 2,                         # 无人机起降对接需要时间
        'PE': 2000,                      # 包裹未送达的惩罚能耗
        'PT': 200,                       # 包裹未送达的惩罚时间
        'PW': 2000,                      # 包裹未送达的惩罚等待时间
        'min_K': 2,                      # 聚类最小K值
        'max_K': 30,                     # 聚类最大K值
        'step': 2,                       # 聚类K值的步长
        'mode': args.mode,               # 用户指定的配送模式（1-能耗，2-时间，3-等待时间）
        'special_packages': args.special_packages  # 用户指定的特殊包裹列表，格式为 [(index, weight), ...]
    }
    return params

def generate_packages(N_packages, Len):
    """
    生成包裹的位置坐标。

    参数:
        N_packages (int): 包裹总数。
        Len (float): 区域边长。

    返回:
        np.ndarray: 包裹的二维坐标数组。
    """
    packages = np.random.rand(N_packages, 2) * Len  # 随机生成包裹坐标，范围在[0, Len)之间
    packages_name = [str(i) for i in range(1, N_packages + 1)]
    print(f"Generated Packages: {packages_name}")  # 打印包裹编号
    return packages

def calculate_manhattan_path_length(path):
    """
    计算给定路径的曼哈顿距离总长度。

    参数:
        path (list of array-like): 路径上的点的列表。

    返回:
        float: 路径的总曼哈顿距离。
    """
    return sum(cityblock(path[i], path[i+1]) for i in range(len(path)-1))

def nearest_neighbor_tsp(points):
    """
    使用最近邻算法求解旅行商问题（TSP），并计算无人车的总路径长度。

    参数:
        points (np.ndarray): 无人车需要访问的点的坐标数组。

    返回:
        tuple:
            list: 无人车路径的点索引列表。
            float: 无人车总行驶距离。
            list: 从起点到每个聚类点的累积距离列表。
    """
    N = len(points)
    visited = [False] * N  # 记录每个点是否已访问
    path = [0]              # 初始路径从第一个点开始
    visited[0] = True      # 标记起点为已访问
    current = 0            # 当前所在点的索引
    distances = []         # 存储每两点之间的欧氏距离
    distances_to_clusters = [0]  # 存储起始点到每个聚类点的累积距离

    for _ in range(1, N):
        # 查找最近的未访问点（使用曼哈顿距离）
        next_point = np.argmin([
            cityblock(points[current], points[j]) if not visited[j] else np.inf
            for j in range(N)
        ])
        # 计算当前点到下一个点的欧氏距离
        distance_to_next_point = np.linalg.norm(points[current] - points[next_point])
        distances.append(distance_to_next_point)  # 存储距离
        distances_to_clusters.append(distances_to_clusters[-1] + distance_to_next_point)  # 更新累积距离
        path.append(next_point)         # 添加到路径
        visited[next_point] = True     # 标记为已访问
        current = next_point            # 更新当前点

    # 计算返回起点的距离
    distances.append(np.linalg.norm(points[current] - points[0]))
    path.append(0)  # 返回起点
    total_car_distance = sum(distances)  # 计算总距离
    return path, total_car_distance, distances_to_clusters

def drone_delivery_paths(relay_point, package_points, special_packages, params):
    """
    计算无人机的配送路径，考虑每次只能配送一个包裹和航程约束。

    参数:
        relay_point (array-like): 当前中继点的坐标。
        package_points (np.ndarray): 当前中继点所属聚类的包裹坐标数组。
        special_packages (list of dict): 特殊包裹的信息列表。
        params (dict): 参数字典。

    返回:
        tuple:
            list of list: 每架无人机的配送路径列表。
            dict: 无人机的飞行距离和往返次数信息。
            list: 无法配送的包裹列表。
    """
    paths = [[] for _ in range(params['N_drones'])]  # 初始化每架无人机的路径列表
    undelivered_packages = []                       # 初始化未配送的包裹列表
    drone_distances = [0] * params['N_drones']      # 记录每架无人机的飞行距离
    drone_round_trips = [0] * params['N_drones']    # 记录每架无人机的往返次数

    # 获取所有特殊包裹的坐标元组列表
    special_indices = [tuple(pkg['point']) for pkg in special_packages]

    # 首先处理特殊包裹，优先配送
    for package in package_points:
        is_special_package = tuple(package) in special_indices
        if is_special_package:
            # 获取当前包裹的权重
            package_info = next(pkg for pkg in special_packages if tuple(pkg['point']) == tuple(package))
            package_weight = package_info['weight']

            # 定义配送路径：中继点 -> 包裹点 -> 中继点
            path_to_package = [relay_point, package, relay_point]
            path_length = calculate_manhattan_path_length(path_to_package)  # 计算路径长度

            if path_length <= params['max_drone_range']:
                # 找到飞行距离最短的无人机
                min_distance_index = np.argmin(drone_distances)
                # 检查无人机是否可以完成此次配送
                if drone_distances[min_distance_index] + path_length * 2 <= params['max_drone_range']:
                    drone_distances[min_distance_index] += path_length * 2  # 更新无人机的飞行距离
                    drone_round_trips[min_distance_index] += 1         # 更新往返次数
                    paths[min_distance_index].append(path_to_package)    # 添加配送路径
                else:
                    undelivered_packages.append(package)  # 无法配送，添加到未配送列表
            else:
                undelivered_packages.append(package)      # 航程不足，添加到未配送列表

    # 处理普通包裹
    for package in package_points:
        is_special_package = tuple(package) in special_indices
        if not is_special_package:
            # 定义配送路径：中继点 -> 包裹点 -> 中继点
            path_to_package = [relay_point, package, relay_point]
            path_length = calculate_manhattan_path_length(path_to_package)
            if path_length <= params['max_drone_range']:
                min_distance_index = np.argmin(drone_distances)
                if drone_distances[min_distance_index] + path_length * 2 <= params['max_drone_range']:
                    drone_distances[min_distance_index] += path_length * 2
                    drone_round_trips[min_distance_index] += 1
                    paths[min_distance_index].append(path_to_package)
                else:
                    undelivered_packages.append(package)
            else:
                undelivered_packages.append(package)

    # 创建包含每架无人机已飞行距离和往返次数的字典
    marked_drone_distances = {'relay_point': relay_point, 'drones': []}
    for i, distance in enumerate(drone_distances):
        marked_drone_distances['drones'].append({
            'drone_id': f'drone_{i}',
            'drone_distance': distance,
            'round_trips': drone_round_trips[i]
        })

    return paths, marked_drone_distances, undelivered_packages

def assign_packages_to_drones(relay_points_drone, packages, distances_to_clusters, kmeans_drone, cluster_special_counts, special_packages, params):
    """
    将包裹分配给无人机，计算无人机的配送路径和未配送包裹。

    参数:
        relay_points_drone (np.ndarray): 所有中继点（包括无人车起点）的坐标数组。
        packages (np.ndarray): 所有包裹的坐标数组。
        distances_to_clusters (list): 起始点到每个聚类点的累积距离列表。
        kmeans_drone (KMeans): KMeans聚类模型。
        cluster_special_counts (dict): 每个聚类中特殊包裹的数量。
        special_packages (list of dict): 特殊包裹的信息列表。
        params (dict): 参数字典。

    返回:
        tuple:
            list of list of list: 每个中继点对应的无人机配送路径列表。
            list of dict: 每个中继点对应的无人机飞行距离和往返次数信息。
            list of array-like: 所有未配送的包裹列表。
            float: 所有中继点到聚类的加权总距离。
    """
    drone_paths = []               # 存储所有中继点的无人机配送路径
    drone_distances = []           # 存储所有中继点的无人机飞行距离信息
    all_undelivered_packages = []  # 存储所有未配送的包裹
    total_distances_to_clusters = 0  # 存储加权后的总距离

    # 遍历所有中继点（从1开始，0为无人车起点）
    for i in range(1, len(relay_points_drone)):
        relay_point = relay_points_drone[i]  # 当前中继点坐标
        cluster_indices = np.where(kmeans_drone.labels_ == i - 1)[0]  # 当前中继点对应聚类的包裹索引
        package_points = packages[cluster_indices]  # 当前聚类的包裹坐标
        num_special_packages_in_cluster = cluster_special_counts.get(i - 1, 0)  # 当前聚类中特殊包裹数量

        if len(package_points) > 0:
            # 计算无人机的配送路径
            drone_paths_for_relay, drone_distances_for_relay, undelivered_packages = drone_delivery_paths(
                relay_point, package_points, special_packages, params
            )
            drone_paths.append(drone_paths_for_relay)                 # 添加无人机配送路径
            drone_distances.append(drone_distances_for_relay)         # 添加无人机飞行距离信息
            all_undelivered_packages.extend(undelivered_packages)     # 添加未配送的包裹

            # 计算加权后的总距离，考虑特殊包裹的权重
            # 计算当前聚类中特殊包裹的总权重
            special_weights_in_cluster = sum(pkg['weight'] for pkg in special_packages if pkg['cluster_index'] == (i - 1))
            total_distances_to_clusters += distances_to_clusters[i] * (
                len(package_points) + special_weights_in_cluster
            )

    return drone_paths, drone_distances, all_undelivered_packages, total_distances_to_clusters

def pick_special_packages(kmeans_drone, packages, relay_points_drone, params):
    """
    根据用户指定的包裹序号和权重选择特殊包裹，并将其分配到对应的聚类中。

    参数:
        kmeans_drone (KMeans): KMeans聚类模型。
        packages (np.ndarray): 所有包裹的坐标数组。
        relay_points_drone (np.ndarray): 所有中继点（包括无人车起点）的坐标数组。
        params (dict): 参数字典。

    返回:
        tuple:
            list of dict: 特殊包裹的信息列表。
            dict: 每个聚类中特殊包裹的数量。
    """
    # 将用户指定的序号和权重转换为0-based索引，并确保索引有效
    special_point_indices = [idx - 1 for idx, _ in params['special_packages']]
    special_point_indices = [idx for idx in special_point_indices if 0 <= idx < len(packages)]

    # 获取特殊包裹的坐标和权重
    special_points = packages[special_point_indices]
    special_weights = [weight for _, weight in params['special_packages']]

    # 预测特殊包裹所属的聚类
    special_clusters = kmeans_drone.predict(special_points)

    cluster_special_counts = {}  # 每个聚类中特殊包裹的数量
    special_packages_info = []    # 特殊包裹的信息列表

    for i, cluster_index in enumerate(special_clusters):
        # 构建特殊包裹的信息字典
        special_packages_info.append({
            'point': special_points[i],
            'weight': special_weights[i],
            'cluster_index': cluster_index,
            'cluster_center': relay_points_drone[cluster_index],
            'original_index': special_point_indices[i] + 1  # 转换回1-based序号
        })
        # 更新每个聚类中特殊包裹的数量
        if cluster_index in cluster_special_counts:
            cluster_special_counts[cluster_index] += special_weights[i]
        else:
            cluster_special_counts[cluster_index] = special_weights[i]

    return special_packages_info, cluster_special_counts

def calculate_total_energy(total_car_distance, drone_paths, params):
    """
    计算总能耗，包括无人车和所有无人机的能耗。

    参数:
        total_car_distance (float): 无人车总行驶距离。
        drone_paths (list of list of list): 所有无人机的配送路径。
        params (dict): 参数字典。

    返回:
        float: 总能耗。
    """
    drone_length = 0  # 无人机总飞行距离
    path_count = 0    # 无人机总起降次数

    # 遍历所有无人机的配送路径，累加飞行距离和路径数量
    for drone_path_group in drone_paths:
        for path_group in drone_path_group:
            for path in path_group:
                drone_path_length = calculate_manhattan_path_length(path)  # 计算单条配送路径的长度
                drone_length += drone_path_length                           # 累加无人机飞行距离
                path_count += 1                                            # 累加配送次数

    # 计算总能耗 = 无人车能耗 + 无人机飞行能耗 + 无人机起降能耗
    total_energy = params['C1'] * total_car_distance + params['C2'] * drone_length + params['C3'] * path_count
    return total_energy

def calculate_total_time(total_car_distance, drone_distances, params):
    """
    计算总时间，包括无人车和所有无人机的配送时间。

    参数:
        total_car_distance (float): 无人车总行驶距离。
        drone_distances (list of dict): 每个中继点对应的无人机飞行距离和往返次数信息。
        params (dict): 参数字典。

    返回:
        float: 总时间。
    """
    # 计算无人车的总时间
    car_time = total_car_distance / params['car_speed']

    # 计算无人机的总时间
    drone_time = 0
    for distances in drone_distances:
        # 对每架无人机，计算其最长的飞行时间（考虑往返次数和起降时间）
        max_drone_distance = max([
            drone['drone_distance'] + drone['round_trips'] * params['SL']
            for drone in distances['drones']
        ])
        drone_time += max_drone_distance / params['drone_speed']  # 累加所有无人机的时间

    # 总时间 = 无人车时间 + 无人机时间
    total_time = car_time + drone_time
    return total_time

def calculate_total_waiting_time(total_distances_to_clusters, drone_paths, special_packages, params):
    """
    计算所有包裹的总等待时间，考虑特殊包裹的权重。

    参数:
        total_distances_to_clusters (float): 所有中继点到聚类点的加权总距离。
        drone_paths (list of list of list): 所有无人机的配送路径。
        special_packages (list of dict): 特殊包裹的信息列表。
        params (dict): 参数字典。

    返回:
        float: 总等待时间。
    """
    drone_time = 0  # 无人机总等待时间

    # 获取特殊包裹的坐标元组列表
    special_indices = [tuple(pkg['point']) for pkg in special_packages]

    # 遍历所有无人机的配送路径，计算等待时间
    for drone_path_group in drone_paths:
        for path_group in drone_path_group:
            for path in path_group:
                package_cords = tuple(path[1])  # 包裹的坐标
                is_special_path = package_cords in special_indices  # 判断是否为特殊包裹
                if is_special_path:
                    # 获取当前包裹的权重
                    package_info = next(pkg for pkg in special_packages if tuple(pkg['point']) == package_cords)
                    weight = package_info['weight']
                else:
                    weight = 1  # 普通包裹的权重为1
                drone_path_length = calculate_manhattan_path_length(path)  # 计算路径长度
                # 计算等待时间：路径长度乘以权重，再除以无人机速度和往返
                drone_time += (drone_path_length * weight) / (params['drone_speed'] * 2)

    # 计算无人车的时间（基于到聚类点的加权总距离）
    car_time = total_distances_to_clusters / params['car_speed']

    # 总等待时间 = 无人车时间 + 无人机等待时间
    total_time = car_time + drone_time
    return total_time

def extract_delivery_order(drone_paths):
    """
    提取所有无人机的包裹配送顺序。

    参数:
        drone_paths (list of list of list): 所有无人机的配送路径。

    返回:
        list of array-like: 包裹的配送顺序列表。
    """
    package_delivery_order = []
    for drone_path_group in drone_paths:
        for path_group in drone_path_group:
            for path in path_group:
                for package in path[1:-1]:  # 排除起始和结束的中继点
                    package_delivery_order.append(package)
    return package_delivery_order

def calculate_metrics(params, total_car_distance, drone_paths, drone_distances, total_distances_to_clusters, undelivered_packages, special_packages):
    """
    根据配送模式计算相应的指标（能耗、时间、等待时间）。

    参数:
        params (dict): 参数字典。
        total_car_distance (float): 无人车总行驶距离。
        drone_paths (list of list of list): 所有无人机的配送路径。
        drone_distances (list of dict): 每个中继点对应的无人机飞行距离和往返次数信息。
        total_distances_to_clusters (float): 所有中继点到聚类点的加权总距离。
        undelivered_packages (list of array-like): 所有未配送的包裹列表。
        special_packages (list of dict): 特殊包裹的信息列表。

    返回:
        tuple:
            float: 能耗（如果模式为1，否则为inf）。
            float: 时间（如果模式为2，否则为inf）。
            float: 等待时间（如果模式为3，否则为inf）。
    """
    if params['mode'] == 1:
        # 模式1：计算总能耗，并考虑未配送包裹的惩罚能耗
        basic_energy = calculate_total_energy(total_car_distance, drone_paths, params)
        penalty_energy = params['PE'] * len(undelivered_packages)
        energy = basic_energy + penalty_energy
        return energy, float('inf'), float('inf')
    elif params['mode'] == 2:
        # 模式2：计算总时间，并考虑未配送包裹的惩罚时间
        basic_time = calculate_total_time(total_car_distance, drone_distances, params)
        penalty_time = params['PT'] * len(undelivered_packages)
        time = basic_time + penalty_time
        return float('inf'), time, float('inf')
    elif params['mode'] == 3:
        # 模式3：计算总等待时间，并考虑未配送包裹的惩罚等待时间
        basic_waiting_time = calculate_total_waiting_time(total_distances_to_clusters, drone_paths, special_packages, params)
        penalty_waiting_time = params['PW'] * len(undelivered_packages)
        waiting_time = basic_waiting_time + penalty_waiting_time
        return float('inf'), float('inf'), waiting_time

def evaluate_clusters(params, packages):
    """
    迭代不同的K值进行聚类，评估并找到最佳的配送方案。

    参数:
        params (dict): 参数字典。
        packages (np.ndarray): 所有包裹的坐标数组。

    返回:
        dict: 包含最佳配送方案的详细信息。
    """
    # 初始化最小值为正无穷，后续会更新为更小的值
    min_energy = float('inf')
    min_time = float('inf')
    min_waiting_time = float('inf')
    best_result = {}  # 存储最佳结果

    # 遍历不同的K值进行聚类评估
    for k in range(params['min_K'], params['max_K'], params['step']):
        print(f"Evaluating K={k}...")  # 输出当前评估的K值

        # 使用KMeans进行聚类
        kmeans_drone = KMeans(n_clusters=k, random_state=42)
        kmeans_drone.fit(packages)
        relay_points_drone = kmeans_drone.cluster_centers_  # 获取聚类中心作为中继点
        relay_points_drone = np.vstack((params['car_start'], relay_points_drone))  # 添加无人车起点

        # 处理特殊包裹（仅在模式3下）
        if params['mode'] == 3 and params['N_special_packages'] > 0:
            special_packages, cluster_special_counts = pick_special_packages(
                kmeans_drone, packages, relay_points_drone, params
            )
        else:
            special_packages = []
            cluster_special_counts = {}

        # 计算无人车路径（使用最近邻算法解决TSP）
        car_path, total_car_distance, distances_to_clusters = nearest_neighbor_tsp(relay_points_drone)

        # 分配包裹给无人机，并获取相关信息
        drone_paths, drone_distances, all_undelivered_packages, total_distances_to_clusters = assign_packages_to_drones(
            relay_points_drone, packages,
            distances_to_clusters, kmeans_drone,
            cluster_special_counts, special_packages, params
        )

        # 提取包裹的配送顺序
        package_delivery_order = extract_delivery_order(drone_paths)

        # 根据配送模式计算相应的指标
        energy, time, waiting_time = calculate_metrics(
            params, total_car_distance, drone_paths, drone_distances,
            total_distances_to_clusters, all_undelivered_packages, special_packages
        )

        # 判断是否为当前最佳结果，并更新
        update_best = False
        if params['mode'] == 1 and energy < min_energy:
            min_energy = energy
            update_best = True
        elif params['mode'] == 2 and time < min_time:
            min_time = time
            update_best = True
        elif params['mode'] == 3 and waiting_time < min_waiting_time:
            min_waiting_time = waiting_time
            update_best = True

        if update_best:
            best_result = {
                'k': k,
                'energy': energy,
                'time': time,
                'waiting_time': waiting_time,
                'car_path': car_path,
                'distances_to_clusters': distances_to_clusters,
                'drone_paths': drone_paths,
                'relay_points_drone': relay_points_drone,
                'undelivered_packages': all_undelivered_packages,
                'special_packages': special_packages,
                'package_delivery_order': package_delivery_order
            }
            print(f"New best result found with K={k}")  # 输出新找到的最佳结果

    return best_result

def visualize_paths(car_path, drone_paths, relay_points_drone, packages, special_packages=None,
                   all_undelivered_packages=None, package_delivery_order=None):
    """
    可视化无人车和无人机的配送路径，以及包裹的位置。

    参数:
        car_path (list): 无人车路径的点索引列表。
        drone_paths (list of list of list): 所有无人机的配送路径。
        relay_points_drone (np.ndarray): 所有中继点（包括无人车起点）的坐标数组。
        packages (np.ndarray): 所有包裹的坐标数组。
        special_packages (list of dict, optional): 特殊包裹的信息列表。
        all_undelivered_packages (list of array-like, optional): 所有未配送的包裹列表。
        package_delivery_order (list of array-like, optional): 包裹的配送顺序列表。
    """
    plt.figure(figsize=(10, 10))  # 创建一个10x10英寸的图形
    plt.title('Delivery Routes')   # 设置图形标题

    # 绘制无人车路径
    plot_path(relay_points_drone, car_path, 'blue', 'Car Path')

    # 绘制无人机路径
    for i, drone_path_group in enumerate(drone_paths):
        for j, path_group in enumerate(drone_path_group):
            for path in path_group:
                drone_path = np.array(path)
                # 仅为路径的第一段添加标签，避免图例重复
                label = f'Drone {i + 1} Path' if j == 0 and i == 0 else ""
                plt.plot(drone_path[:, 0], drone_path[:, 1], 'cyan', linestyle='--', label=label)

    # 绘制所有包裹点
    for i, package in enumerate(packages):
        plt.scatter(package[0], package[1], c='red', label='Packages' if i == 0 else "")
        if package_delivery_order is not None:
            plt.text(package[0], package[1], str(i + 1), fontsize=9, ha='right')  # 添加包裹编号标签

    # 绘制特殊包裹点
    if special_packages:
        for i, package in enumerate(special_packages):
            plt.scatter(package['point'][0], package['point'][1], c='yellow', marker='*',
                        label='Special Packages' if i == 0 else "")

    # 绘制未送达包裹点
    if all_undelivered_packages:
        for i, package in enumerate(all_undelivered_packages):
            plt.scatter(package[0], package[1], c='black', marker='x', label='Undelivered Packages' if i == 0 else "")

    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()      # 显示图形

def plot_path(points, path, color, label):
    """
    绘制给定路径的连线和节点。

    参数:
        points (np.ndarray): 所有点的坐标数组。
        path (list): 路径的点索引列表。
        color (str): 连线和节点的颜色。
        label (str): 图例标签。
    """
    path_points = points[path]  # 获取路径上的所有点的坐标
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        # 仅为路径的第一段添加标签，避免图例重复
        plt.plot([x1, x2], [y1, y2], color=color, label=label if i == 0 else "")
    plt.scatter(path_points[:, 0], path_points[:, 1], c=color)  # 绘制路径上的所有点

def main(args=None):
    """
    主函数，负责解析命令行参数，设置参数，生成包裹，评估不同K值的聚类方案，找到最佳方案并进行可视化。
    """
    # 如果没有提供参数，使用sys.argv
    if args is None:
        args = sys.argv[1:]

    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="无人车与无人机配送路径优化程序")
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=3,
                        help='配送模式：1-能耗，2-时间，3-等待时间（包含特殊点）')
    parser.add_argument('--special_packages', type=parse_special_package, nargs='*', default=[],
                        help='特殊包裹的序号和权重列表（例如：--special_packages 10:3 32:5 59:1）')
    parsed_args = parser.parse_args(args)

    # 设置参数
    params = set_parameters(parsed_args)

    # 生成包裹位置
    packages = generate_packages(params['N_packages'], params['Len'])

    # 验证特殊包裹序号是否有效
    if params['N_special_packages'] > 0:
        # 检查是否有包裹序号超出范围
        invalid_indices = [idx for idx, _ in params['special_packages'] if idx < 1 or idx > params['N_packages']]
        if invalid_indices:
            print(f"Error: 特殊包裹序号超出范围：{invalid_indices}。请确保序号在1到{params['N_packages']}之间。")
            return

    # 评估不同的K值并找到最佳结果
    best_result = evaluate_clusters(params, packages)

    # 检查是否找到有效的聚类结果
    if not best_result:
        print("No valid clustering found based on the given parameters.")
        return

    # 输出最佳结果
    print(f"\n最佳结果:")
    print(f"最佳 K 值: {best_result['k']}")
    if params['mode'] == 1:
        print(f"最低能耗: {best_result['energy']}")
    elif params['mode'] == 2:
        print(f"最低时间: {best_result['time']}")
    elif params['mode'] == 3:
        print(f"最低等待时间: {best_result['waiting_time']}")

    # 可视化最佳配送路径
    visualize_paths(
        best_result['car_path'],
        best_result['drone_paths'],
        best_result['relay_points_drone'],
        packages,
        best_result['special_packages'],
        best_result['undelivered_packages'],
        best_result['package_delivery_order']
    )



if __name__ == '__main__':
    # 示例：在代码内部传递参数
    test_args = ['--mode', '3', '--special_packages', '10:2', '32:5', '59:5']
    main(test_args)
