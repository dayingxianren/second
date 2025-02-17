import matplotlib.pyplot as plt
import numpy as np
from data import Data

class Draw:
    def __init__(self):
        self.data = Data()

    # 可视化路径
    def plot_path(self, points, path, color, label):
        path_points = points[path]
        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i + 1]
            plt.plot([x1, x2], [y1, y1], color=color, label=label if i == 0 else "")
            plt.plot([x2, x2], [y1, y2], color=color)
        plt.scatter(path_points[:, 0], path_points[:, 1], c=color)

    def visualize_paths(self, car_path, drone_paths, relay_points_drone, packages, special_packages=None,
                        all_undelivered_packages=None):
        plt.figure(figsize=(10, 5))
        # 无人车+无人机方案
        plt.subplot(1, 2, 1)
        plt.title('UAV+UGV')
        self.plot_path(relay_points_drone, car_path, 'blue', 'Car Path')
        for i, drone_path_group in enumerate(drone_paths):
            for j, path_group in enumerate(drone_path_group):
                for path in path_group:
                    drone_path = np.array(path)
                    plt.plot(drone_path[:, 0], drone_path[:, 1], color='cyan', linestyle='--')
        # 显示包裹位置和无法配送的包裹
        special_indices = ([point['original_index'] for point in special_packages]if self.data.mode == 3 else None)  # 获取特殊点的索引列表
        if special_indices is not None:
            special_points = packages[special_indices]
            plt.scatter(special_points[:, 0], special_points[:, 1], c='yellow', label='Special Packages')
        # 绘制非特殊点的包裹
        non_special_indices = ([i for i in range(len(packages)) if i not in special_indices] if self.data.mode == 3 else [i for i in range(len(packages))])
        non_special_points = packages[non_special_indices]
        plt.scatter(non_special_points[:, 0], non_special_points[:, 1], c='red', label='Packages')

        if all_undelivered_packages:
            undelivered_packages = np.array(all_undelivered_packages)
            plt.scatter(undelivered_packages[:, 0], undelivered_packages[:, 1], c='black', marker='x',
                        label='Undelivered Packages')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 输出无法配送的包裹信息
        if all_undelivered_packages:
            print(f"Number of undelivered packages: {len(all_undelivered_packages)}")
            print(f"Undelivered packages: {all_undelivered_packages}")