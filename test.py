import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rice, norm

# ------------------- 环境参数 -------------------
d_BS = 200  # 邻居基站间距 (m) —— 备用参数
d_BS0_BSi = 140  # BS0 到任一邻居基站的距离 (m)
d_BS0_UE = 70    # BS0 到 UE 的距离 (m)
# 假设 UE 与干扰 BS 在同一直线上，则邻居 BS 到 UE 的距离为：
d_BSi_UE = np.sqrt(10900)  # = 70 m

num_interfering_BS = 3  # 邻居基站数量

P_BS_base_dBm = 40  # 所有基站的固定发射功率 (dBm)
noise_power_dBm = -95  # UE 接收的噪声功率 (dBm)
SINR_target_range = [4, 5]  # 目标 SINR 阈值范围 (dB)

# ------------------- 路径损耗模型参数 -------------------
d_r = 1  # 参考距离 (m)
kappa_r_dB = 43.3  # 参考距离 d_r 处的路径损耗 (dB)
M = 2  # LOS 场景下的路径损耗指数
sigma_xi = 3  # 阴影衰落标准差 (dB)

# ------------------- Rician 衰落模型参数 -------------------
K = 3  # Rician K 因子
s = np.sqrt(K / (K + 1))         # 直射路径强度
sigma = np.sqrt(1 / (2 * (K + 1))) # 散射信号标准差

# ------------------- 计算路径损耗 -------------------
def compute_path_loss(d):
    # 计算指定距离 d 的路径损耗，包含阴影衰落
    shadowing_dB = norm.rvs(0, sigma_xi)  # 阴影衰落
    path_loss_dB = 43.3 + 20 * np.log10(d / d_r) + shadowing_dB
    return path_loss_dB

# 计算 BS0 到 UE 的路径损耗（使用 d_BS0_UE = 70 m）
L_BS0_UE_dB = compute_path_loss(d_BS0_UE)

# ------------------- 计算 Rician 小尺度衰落 -------------------
def generate_rician_fading():
    # 生成符合 Rician 分布的信道增益，并转换为 dB 形式（10*log10(|h|^2)）
    fading = rice.rvs(s / sigma, scale=sigma)
    return 10 * np.log10(fading ** 2)

# 计算 BS0 对 UE 的 Rician 衰落
fading_BS0_UE_dB = generate_rician_fading()

# ------------------- 计算接收信号功率 -------------------
# BS0 对 UE 的接收功率（dBm）计算公式：
# P_rx = P_tx - L_pathloss + L_fading
P_received_BS0_UE_dBm = P_BS_base_dBm - L_BS0_UE_dB + fading_BS0_UE_dB

# ------------------- 计算所有邻居基站对 UE 的总干扰 -------------------
# 注意：干扰信号需要计算的是邻居 BS 到 UE 的路径损耗，使用 d_BSi_UE（70 m）
P_interference_mW = 0  # 初始化总干扰功率 (mW)
for _ in range(num_interfering_BS):
    # 对于每个邻居 BS，计算其到 UE 的路径损耗（使用 d_BSi_UE）
    L_BSi_UE_dB = compute_path_loss(d_BSi_UE)
    # 计算该邻居 BS 的 Rician 衰落（小尺度）
    fading_BSi_UE_dB = generate_rician_fading()
    # 计算该邻居 BS 对 UE 的接收功率（dBm）
    P_interfering_BSi_dBm = P_BS_base_dBm - L_BSi_UE_dB + fading_BSi_UE_dB
    # 转换到 mW 累加
    P_interference_mW += 10 ** (P_interfering_BSi_dBm / 10)

# 将总干扰功率转换回 dBm
P_interference_total_dBm = 10 * np.log10(P_interference_mW)

# ------------------- 计算 SINR -------------------
def compute_SINR(P_signal_dBm, P_interference_dBm, noise_dBm):
    # 将 dBm 转换为 mW 进行加法计算，然后再转换回 dB
    P_signal_mW = 10 ** (P_signal_dBm / 10)
    P_interference_mW = 10 ** (P_interference_dBm / 10)
    noise_mW = 10 ** (noise_dBm / 10)
    SINR_mW = P_signal_mW / (noise_mW + P_interference_mW)
    print(f"UE 处的 SINR_mw: {SINR_mW:.2f} ")
    return 10 * np.log10(SINR_mW)

SINR_dB = compute_SINR(P_received_BS0_UE_dBm, P_interference_total_dBm, noise_power_dBm)

# ------------------- 打印结果 -------------------
print(f"BS0 到 UE 的接收功率: {P_received_BS0_UE_dBm:.2f} dBm")
print(f"所有邻居基站对 UE 的总干扰功率: {P_interference_total_dBm:.2f} dBm")
print(f"UE 处的 SINR: {SINR_dB:.2f} dB")

# ------------------- 结果可视化 -------------------
plt.bar(['Received Power (BS0->UE)', 'Total Interference Power', 'Noise Power'],
        [P_received_BS0_UE_dBm, P_interference_total_dBm, noise_power_dBm],
        color=['blue', 'red', 'green'])
plt.axhline(y=SINR_target_range[0], color='black', linestyle='--', label='SINR Target Lower Bound')
plt.axhline(y=SINR_target_range[1], color='gray', linestyle='--', label='SINR Target Upper Bound')
plt.title("Power Levels and SINR at UE")
plt.ylabel("Power (dBm)")
plt.legend()
plt.grid()
plt.show()
