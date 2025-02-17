import numpy as np
from data import Data

class Packages:
    def __init__(self):
        self.data = Data()

    def generate_packages(self):
        np.random.seed()
        # 生成包裹位置（含小数）
        packages = np.random.rand(self.data.N_packages, 2) * self.data.Len
        return packages