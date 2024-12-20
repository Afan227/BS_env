# 1. 基础参数设置
AREA_SIZE = 5000  # 区域边长 (单位：米)
NUM_BASE_STATIONS = 5  # 基站数量
NUM_USERS = 50  # 用户数量
TX_POWER = 44  # 基站发射功率 (单位：dBm)  25W
NOISE_POWER = -100  # 热噪声功率 (单位：dBm)
PATH_LOSS_EXPONENT = 3  # 路径损耗指数
SHADOWING_STD = 8  # 阴影衰落标准差 (单位：dB)
MESH_NUM = 1000   # 生成地图的网格精细程度
D_BP = 1500   # 参考距离
H_BS = 25 # 基站高度
H_UE = 1.5 # 用户高度 m
F = 2  # GHz