# import numpy as np
# import matplotlib.pyplot as plt
#
# # 1. 基础参数设置
# AREA_SIZE = 1000  # 区域边长 (单位：米)
# NUM_BASE_STATIONS = 5  # 基站数量
# NUM_USERS = 50  # 用户数量
# TX_POWER = 40  # 基站发射功率 (单位：dBm)
# NOISE_POWER = -100  # 热噪声功率 (单位：dBm)
# PATH_LOSS_EXPONENT = 3.5  # 路径损耗指数
# SHADOWING_STD = 8  # 阴影衰落标准差 (单位：dB)
#
# # 功能函数
# def dbm_to_watt(dbm):
#     """将 dBm 转换为瓦特"""
#     return 10 ** ((dbm - 30) / 10)
#
# def watt_to_dbm(watt):
#     """将瓦特转换为 dBm"""
#     return 10 * np.log10(watt) + 30
#
# def path_loss(distance, exponent):
#     """计算路径损耗 (单位：dB)"""
#     return 10 * exponent * np.log10(distance + 1e-9)
#
# # 2. 初始化基站和用户分布
# np.random.seed(42)
# base_stations = np.random.uniform(0, AREA_SIZE, (NUM_BASE_STATIONS, 2))
# users = np.random.uniform(0, AREA_SIZE, (NUM_USERS, 2))
#
# # 3. 信道模型计算
# def calculate_sinr(user_pos, base_stations, tx_power, noise_power):
#     distances = np.linalg.norm(base_stations - user_pos, axis=1)  # 计算用户到各基站的距离
#     path_losses = path_loss(distances, PATH_LOSS_EXPONENT)  # 路径损耗
#     received_powers = dbm_to_watt(tx_power - path_losses)  # 接收功率
#     signal_power = np.max(received_powers)  # 最大接收功率作为信号
#     interference_power = np.sum(received_powers) - signal_power  # 干扰功率
#     noise_power_watt = dbm_to_watt(noise_power)
#     sinr = signal_power / (interference_power + noise_power_watt)  # SINR 计算
#     return watt_to_dbm(sinr)
#
# # 4. SINR 计算和可视化
# sinr_values = []
# for user_pos in users:
#     sinr = calculate_sinr(user_pos, base_stations, TX_POWER, NOISE_POWER)
#     sinr_values.append(sinr)
#
# # 可视化用户、基站和 SINR
# plt.figure(figsize=(10, 8))
# plt.scatter(base_stations[:, 0], base_stations[:, 1], c='red', label='Base Stations', s=100)
# plt.scatter(users[:, 0], users[:, 1], c=sinr_values, cmap='viridis', label='Users')
# plt.colorbar(label='SINR (dBm)')
# plt.title('Base Station and User Distribution with SINR')
# plt.xlabel('X-coordinate (m)')
# plt.ylabel('Y-coordinate (m)')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # 5. 优化目标示例：调整基站位置以最大化平均 SINR
# from scipy.optimize import minimize
#
# def optimize_base_station_positions(base_stations, users, tx_power, noise_power):
#     def objective(positions):
#         positions = positions.reshape(NUM_BASE_STATIONS, 2)
#         avg_sinr = 0
#         for user_pos in users:
#             sinr = calculate_sinr(user_pos, positions, tx_power, noise_power)
#             avg_sinr += sinr
#         return -avg_sinr / len(users)  # 目标是最大化平均 SINR，因此取负值
#
#     initial_positions = base_stations.flatten()
#     result = minimize(objective, initial_positions, method='L-BFGS-B', bounds=[(0, AREA_SIZE)] * len(initial_positions))
#     return result.x.reshape(NUM_BASE_STATIONS, 2)
#
# optimized_positions = optimize_base_station_positions(base_stations, users, TX_POWER, NOISE_POWER)
#
# # 可视化优化后结果
# plt.figure(figsize=(10, 8))
# plt.scatter(optimized_positions[:, 0], optimized_positions[:, 1], c='red', label='Optimized Base Stations', s=100)
# plt.scatter(users[:, 0], users[:, 1], c=sinr_values, cmap='viridis', label='Users')
# plt.colorbar(label='SINR (dBm)')
# plt.title('Optimized Base Station Positions')
# plt.xlabel('X-coordinate (m)')
# plt.ylabel('Y-coordinate (m)')
# plt.legend()
# plt.grid(True)
# plt.show()


# 导入库
import osmnx as ox
print(ox.__version__)
from IPython.display import Image
# 可选，储存图片路径
img_folder = "images"
extension = "png"
size = 480 #图片长宽大小


# 定义函数
def make_plot(place, point,
              network_type="drive",
              dpi=80, dist=1000, default_width=4,
              street_widths=None):  # dists 填入米
    tags = {"building": True}
    fp = f"./{img_folder}/{place}.{extension}"  # 图片的地址保存
    gdf = ox.geometries_from_point(point, tags, dist=dist)
    fig, ax = ox.plot_figure_ground(
        point=point,
        dist=dist,
        network_type=network_type,
        default_width=default_width,
        street_widths=street_widths,
        save=False,
        show=False,
        close=True,
    )
    fig, ax = ox.plot_footprints(
        gdf, ax=ax, filepath=fp, dpi=dpi, save=True, show=False, close=True
    )
    return Image(fp, height=size, width=size)


# 执行函数
place = "shanghai"
point = (31.238850562378246, 121.48578354186554)  # 填入wgs1984坐标
make_plot(place, point, network_type="all", default_width=2, street_widths={"primary": 6})





