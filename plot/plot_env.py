import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
def plot_env(x_grid,y_grid,sinr_grid):
    # 设置一个阈值，例如 threshold_value
    threshold_value = 4  # 小于此值为一种颜色，大于此值为另一种颜色

    # 设置颜色区间
    levels = [sinr_grid.min(), threshold_value, sinr_grid.max()]  # 分为三个区间

    # 使用 BoundaryNorm 来规范化颜色
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)

    # 自定义颜色映射：从浅蓝色到白色
    colors = [(0, 'lightgrey'), (1, 'green')]  # 从白色到浅蓝色
    n_bins = 100  # 颜色过渡的阶数
    cmap_name = 'light_blue_white'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # 绘制等高线图，使用自定义颜色映射
    contour = plt.contourf(x_grid, y_grid, sinr_grid, levels=levels, cmap=custom_cmap, norm=norm)

    # 添加 colorbar
    plt.colorbar(contour, label='SINR (dB)')

    # 添加图标标签和标题
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('SINR Distribution over Area')

    # 显示图形
    plt.show()

def plot_reward(history_rewards):
    # 示例：假设 episode_rewards 是历史 episode 的收益

    # 构造 x 轴数据（episode 序号）
    episodes = np.arange(1, len(history_rewards) + 1)

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, history_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reinforcement Learning Episode Rewards')
    plt.grid(True)
    plt.show()