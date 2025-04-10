import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
from matplotlib.colors import LogNorm, Normalize

# ==============================
# 参数设置（符合标准体育场布局）
# ==============================
total_time = 4 * 60  # 总时长：4小时（16:00-20:00）
time_step = 5  # 时间步长：5分钟
num_frames = total_time // time_step

# 场地尺寸（国际足联标准足球场尺寸参考）
x_min, x_max = 0, 50  # 长度：500米（含看台）
y_min, y_max = 0, 30  # 宽度：300米（含看台）

# 高斯分量参数（分西、东、南、北四个观众席区域）
K = 5  # 8个高斯分量

params = {
    # 西区（原x=50→6.25，y=100→25）
    0: {"N_max": 40, "sigma": [20, 7], "mu": [15, 5], "type": "west"},
    1: {"N_max": 40, "sigma": [20, 7], "mu": [35, 5], "type": "west"},

    # 东区（原x=350→43.75，y=100→25）
    2: {"N_max": 80, "sigma": [20, 7], "mu": [25, 25], "type": "east"},

    # 南区（原x=200→25，y=30→7.5）
    3: {"N_max": 80, "sigma": [7, 25], "mu": [5, 15], "type": "east"},

    # 北区（原x=200→25，y=170→42.5）
    4: {"N_max": 80, "sigma": [7, 25], "mu": [45, 15], "type": "east"},

}


# ==============================
# 动态分布生成（考虑四周看台）
# ==============================
def generate_distribution(t_minutes):
    xx, yy = np.meshgrid(np.linspace(0, 50, 500),
                         np.linspace(0, 30, 300))
    grid = np.dstack((xx, yy))
    density = np.zeros_like(xx)

    t_hours = t_minutes / 60
    phase = "pre" if t_hours < 1 else ("game" if t_hours < 3 else "post")

    for k in range(K):
        p = params[k]

        # 动态参数计算
        if phase == "pre":
            N_k = p["N_max"] * (t_hours / 1)
            sigma = p["sigma"]
            mu = p["mu"]
        elif phase == "game":
            N_k = p["N_max"]
            sigma = [s * 0.8 for s in p["sigma"]]  # 比赛期间更集中
            mu = p["mu"]
        else:
            N_k = p["N_max"] * max(0, (1 - (t_hours - 3)))
            sigma = [s * (1 + 0.5 * (t_hours - 3)) for s in p["sigma"]]
            mu = p["mu"]

        # 生成各向异性分布
        cov = [[sigma[0] ** 2, 0], [0, sigma[1] ** 2]]
        rv = multivariate_normal(mu, cov)
        pdf_values = rv.pdf(grid)

        # 计算该分量的总积分（用于归一化）
        integral = pdf_values.sum() * (xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0])  # 网格面积积分
        density_k = (N_k / integral) * pdf_values  # 确保积分总人数为N_k

        # 看台区域密度增强
        if p["type"] in ["west", "east", "south", "north"]:
            density += 3.0 * density_k
        else:
            density += density_k

    return xx, yy, density


# ==============================
# 可视化增强（显示四周看台结构）
# ==============================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("X Position (m)", fontsize=10)
ax.set_ylabel("Y Position (m)", fontsize=10)
title = ax.set_title("Stadium Crowd Distribution", fontsize=14)

# # 绘制场地结构
# ax.plot([100, 100], [50, 250], 'w--', lw=1, alpha=0.6)  # 西看台边界
# ax.plot([400, 400], [50, 250], 'w--', lw=1, alpha=0.6)  # 东看台边界
# ax.plot([150, 350], [40, 40], 'w--', lw=1, alpha=0.6)  # 南看台边界
# ax.plot([150, 350], [260, 260], 'w--', lw=1, alpha=0.6)  # 北看台边界

# 初始密度图
xx, yy, density = generate_distribution(0)
heatmap = ax.pcolormesh(xx, yy, density, shading='auto',
                        cmap='inferno',
                        norm=Normalize(vmin=0, vmax=3))  # LogNorm范围1~36

# 创建colorbar
cbar = plt.colorbar(heatmap, ax=ax, pad=0.02, shrink=0.6, aspect=20)


ax.set_aspect("equal")

# 标签样式
cbar.set_label('Density (users/m²)', fontsize=10)
cbar.ax.tick_params(labelsize=8, colors='black')


def update(frame):
    t_minutes = frame * time_step
    current_time = 16 + t_minutes // 60 + (t_minutes % 60) / 60
    xx, yy, density = generate_distribution(t_minutes)

    heatmap.set_array(density.ravel())
    title.set_text(f"Time: {int(current_time // 1):02d}:{int((current_time % 1) * 60):02d}")
    time.sleep(0.2)
    return heatmap, title


ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
plt.show()
