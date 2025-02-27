import pandas as pd
import os
import numpy as np

from scipy.spatial import Delaunay
# 1. 基础参数设置
AREA_SIZE = 200  # 区域边长 (单位：米)
NUM_BASE_STATIONS = 5  # 基站数量
NUM_USERS = 50  # 用户数量
NUM_BUILDING = 70 # 建筑物数量
TX_POWER_S = 40  # 基站发射功率 (单位：dBm)  25W
NOISE_POWER = -95  # 热噪声功率 (单位：dBm)
PATH_LOSS_EXPONENT = 2  # 路径损耗指数
SHADOWING_STD = 3  # 阴影衰落标准差 (单位：dB)
MESH_NUM = 1000   # 生成地图的网格精细程度
D_BP = 3200   # 参考距离
H_BS = 25 # 基站高度
H_UE = 1.5 # 用户高度 m
F = 2  # GHz

iter_number = 500
action_space = [37,38,39,40,41,42,43]

# 采样设置
# 设定采样点间隔为2米
sampling_interval = 2

# 生成200x200区域内的采样点坐标 (每隔2米一个采样点)
x_sampled = np.arange(0, 200, sampling_interval)  # 从0到200，间隔2米
y_sampled = np.arange(0, 200, sampling_interval)  # 从0到200，间隔2米

# 使用 meshgrid 创建二维网格（所有采样点坐标）
x_sampled, y_sampled = np.meshgrid(x_sampled, y_sampled)

# 展开为一维数组，以便传递给get_sinr函数
x_sampled = x_sampled.flatten()
y_sampled = y_sampled.flatten()
z_sampled = np.zeros(len(x_sampled))
# 使用 get_sinr 函数获取每个采样点的 SINR 值
# 构造二维采样点数组
points = np.column_stack((x_sampled, y_sampled))
# 预计算三角剖分
tri = Delaunay(points)

script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录

file_path_user = os.path.join(script_dir, 'user_position.xlsx')   # 拼接文件路径'user_position.xlsx'
file_path_building = os.path.join(script_dir, 'building_position.xlsx')
file_path_bs = os.path.join(script_dir, 'bs_position.xlsx')

df = pd.read_excel(file_path_user)
user = list(zip(df['x'], df['y'], df['H_UE']))

building = pd.read_excel(file_path_building).to_dict(orient='records')

df = pd.read_excel(file_path_bs)
bs_position = list(zip(df['x'], df['y'], df['H_BS']))
