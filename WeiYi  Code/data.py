import numpy as np

class Data:
    def __init__(self):
        self.car_start = np.array([0, 0])    # 无人车起点
        self.car_speed = 1                   # 无人车速度：每单位时间1个网格单位
        self.drone_speed = 2                 # 无人机速度：每单位时间2个网格单位
        self.N_packages = 100                # 包裹总数
        self.N_special_packages = 10         # 特殊包裹总数
        self.Len = 100                       # 区域的边长
        self.N_drones = 5                    # 无人机数量
        self.max_drone_range = 100  # 无人机的最大飞行航程（曼哈顿距离）
        # 能耗
        self.C1 = 20  # 无人车的能耗参数
        self.C2 = 5  # 无人机飞行能耗参数
        self.C3 = 10  # 无人机起降能耗参数
        # 对接时间
        self.SL = 2  # 无人机起降对接需要时间
        # 惩罚
        self.PE = 2000  # 包裹未送达的惩罚能耗
        self.PT = 200  # 包裹未送达的惩罚时间
        self.PW = 2000  # 包裹未送达的惩罚等待时间
        # 等待时间
        self.SP = 2  # 特殊点等待时间的权重
        self.SW = 1  # 普通点等待时间的权重
        # 聚类
        self.min_K = 2  # 聚类最小值
        self.max_K = 30  # 聚类最大值
        self.step = 2  # 聚类数值跨度
        # 类型
        self.mode = 2  # 价值函数的类型（1：能耗；2：时间；3：等待时间（包含特殊点））

    def change_car_speed(self):
        car_speed = np.random.randint(1, 3)
        return car_speed