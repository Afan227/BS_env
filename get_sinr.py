import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 假设这是你已有的 get_sinr 函数
def get_sinr(x, y):
    # 这里是一个示例，你需要根据实际情况实现 SINR 计算
    return np.random.uniform(-10, 30)  # 假设SINR值在 0 到 30 dB 之间

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

# 使用 get_sinr 函数获取每个采样点的 SINR 值
sinr_values = np.array([get_sinr(x, y) for x, y in zip(x_sampled, y_sampled)])

# 创建一个更细的网格，覆盖整个200x200的区域（例如2000x2000网格）
x_grid, y_grid = np.meshgrid(np.linspace(0, 200, 2000), np.linspace(0, 200, 2000))

# 使用SciPy的griddata进行插值，采用'linear'插值方法
sinr_grid = griddata((x_sampled, y_sampled), sinr_values, (x_grid, y_grid), method='linear')

# 查找 NaN 值的位置
nan_mask = np.isnan(sinr_grid)

# 仅对NaN值的位置进行填充，使用 'nearest' 方法进行填充
sinr_grid[nan_mask] = griddata((x_sampled, y_sampled), sinr_values, (x_grid[nan_mask], y_grid[nan_mask]), method='nearest')
print(griddata((x_sampled, y_sampled),sinr_values,(5.432413,6.341332),method='linear'))
# 可视化结果
plt.contourf(x_grid, y_grid, sinr_grid, 20, cmap='viridis')
plt.colorbar(label='SINR (dB)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('SINR Distribution over 200x200m area')
plt.show()
