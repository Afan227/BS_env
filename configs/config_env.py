import pandas as pd
import os
# 1. 基础参数设置
AREA_SIZE = 5000  # 区域边长 (单位：米)
NUM_BASE_STATIONS = 5  # 基站数量
NUM_USERS = 50  # 用户数量
NUM_BUILDING = 70 # 建筑物数量
TX_POWER = 45  # 基站发射功率 (单位：dBm)  25W
NOISE_POWER = -100  # 热噪声功率 (单位：dBm)
PATH_LOSS_EXPONENT = 3  # 路径损耗指数
SHADOWING_STD = 8  # 阴影衰落标准差 (单位：dB)
MESH_NUM = 1000   # 生成地图的网格精细程度
D_BP = 320   # 参考距离
H_BS = 25 # 基站高度
H_UE = 1.5 # 用户高度 m
F = 2  # GHz

script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录

file_path_user = os.path.join(script_dir, 'user_position.xlsx')   # 拼接文件路径'user_position.xlsx'
file_path_building = os.path.join(script_dir, 'building_position.xlsx')
file_path_bs = os.path.join(script_dir, 'bs_position.xlsx')

df = pd.read_excel(file_path_user)
user = list(zip(df['x'], df['y'], df['H_UE']))

building = pd.read_excel(file_path_building).to_dict(orient='records')

df = pd.read_excel(file_path_bs)
bs = list(zip(df['x'], df['y'], df['H_BS']))
