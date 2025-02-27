from scipy.interpolate import LinearNDInterpolator
import gymnasium as gym
from gymnasium.spaces import Discrete
from scipy.stats import rice
from configs.config_env import *
from scipy.interpolate import griddata
GRID_NOT_GENERATED_FLAG = -1
import time
# np.random.seed(100)
# We only consider Y is a rectangular in R^d
class BaseStationEnv(gym.Env):
    def __init__(self, area_size=AREA_SIZE, num_base_stations=NUM_BASE_STATIONS,
                 num_users=NUM_USERS, tx_power=TX_POWER_S, noise_power=NOISE_POWER,shadowing_std = SHADOWING_STD):
        super(BaseStationEnv, self).__init__()

        # 环境参数
        self.area_size = area_size  # 区域大小
        self.num_base_stations = num_base_stations  # 基站数量
        self.num_users = num_users  # 用户数量
        self.tx_power = np.array([tx_power,tx_power,tx_power,tx_power,tx_power])  # 基站发射功率 (dBm)
        self.noise_power = noise_power  # 热噪声功率 (dBm)
        self.path_loss_exponent = PATH_LOSS_EXPONENT  # 路径损耗指数
        self.shadowing_std = shadowing_std  # 阴影衰落标准差 (dB)
        # 状态空间 (基站和用户的位置)
        self.state = None

        # 动作空间 (基站位置调整)
        self.action_space = self.action_space = Discrete(7)

        # 基站位置
        self.bs = np.array(bs_position)
        # 将存储的大尺度衰落转化为数组
        self.large_scale = np.array(self.calculate_large_scale())


    def reset(self, seed = None,options = None):
        """重置环境"""

        self.state = self.sinr_map()

        return self.state

    def step(self, action):
        """环境状态更新"""
        # 更新基站位置
        action = action_space[action]
        print(f'本次选择的动作为：{action}')
        self.tx_power[4] = action
        #self.tx_power[4] = np.clip(self.tx_power[4], 37, 43)  # 限制在基站发射功率内
        # 计算奖励 (以平均 SINR 为奖励)
        sinr_map = self.sinr_map()
        cost = self.calculate_cost(sinr_map[2])
        power_consumption = dbm_to_watt(self.tx_power[4])
        reward = self.calculate_reward(sinr_map[2],power_consumption)
        # 终止条件 (这里假设单步不终止，可根据实际需求调整)
        done = False

        return sinr_map, reward, cost ,done, {}


    def render(self, mode='human'):
        """可视化环境"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.scatter(self.users[:, 0], self.users[:, 1], c='blue', label='Users')
        plt.scatter(self.base_stations[:, 0], self.base_stations[:, 1], c='red', label='Base Stations', s=100)
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        plt.title('Base Station Deployment')
        plt.xlabel('X-coordinate (m)')
        plt.ylabel('Y-coordinate (m)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sinr_map(self):
        sinr_values = self.calculate_sinr(x_sampled, y_sampled, z_sampled)
        # 创建一个更细的网格，覆盖整个200x200的区域（例如2000x2000网格）
        x_grid, y_grid = np.meshgrid(np.linspace(0, 200, 2000), np.linspace(0, 200, 2000))

        # 使用SciPy的griddata进行插值，采用'linear'插值方法

        interp = LinearNDInterpolator(tri, sinr_values)
        sinr_grid = interp((x_grid, y_grid))
        #sinr_grid = griddata((x_sampled, y_sampled), sinr_values, (x_grid, y_grid), method='linear')

        # 查找 NaN 值的位置
        nan_mask = np.isnan(sinr_grid)

        # 仅对NaN值的位置进行填充，使用 'nearest' 方法进行填充
        sinr_grid[nan_mask] = griddata((x_sampled, y_sampled), sinr_values, (x_grid[nan_mask], y_grid[nan_mask]),
                                       method='nearest')


        return x_grid, y_grid, sinr_grid   # 2000*2000


    def calculate_large_scale(self):

        # 存储每个采样点的大尺度衰落（路径损耗和阴影衰落）
        large_scales = []
        for x, y, z in zip(x_sampled, y_sampled, z_sampled):
            distances = np.linalg.norm(self.bs - [x, y, z], axis=1)
            large_scale = self.pathloss_los(distances, 3)
            large_scales.append(large_scale)

        return large_scales

    def pathloss_los(self,distance, shadowing_std):
        PL = 43.3 + 20 * np.log10(distance)
        shadowing = np.random.normal(0, shadowing_std , size=distance.shape)
        PL += shadowing
        return PL

    def calculate_sinr(self,x,y,z):
        """计算所有用户的平均 SINR，包括阴影衰落"""

        N_samples = len(x)
        N_bs = self.num_base_stations


        def dbm_to_watt(dbm):
            return 10 ** ((dbm - 30) / 10)

        # Rician衰落模型：LOS下的信号衰落
        def rician_fading():
            K = 3  # Rician K 因子
            s = np.sqrt(K / (K + 1))  # 直射路径强度
            sigma = np.sqrt(1 / (2 * (K + 1)))  # 散射信号标准差
            # 生成瑞利衰落的随机部分（代表多径部分）
            fading = rice.rvs(s / sigma, scale=sigma,size=(N_samples, N_bs))
            return 10 * np.log10(fading ** 2)

        def path_loss():
            # 添加阴影衰落,添加 LOS 与 NLOS 的不同计算方法
            """
            PL = 43.3 + 20 * np.log10(distance)
            """
            PL = self.large_scale - rician_fading()
            return  PL

        path_losses= path_loss()  # 包含阴影衰落
        received_powers = dbm_to_watt(self.tx_power - path_losses)

        signal_power = np.max(received_powers,axis=1)
        index_of_signal_power = np.argmax(received_powers,axis=1)

        interference_power = np.sum(received_powers,axis=1) - signal_power

        noise_power_watt = dbm_to_watt(self.noise_power)
        sinr = signal_power / (interference_power + noise_power_watt)
        sinr_db = 10*np.log10(sinr)
        #c = np.log2(1+ sinr)/10

        return sinr_db

    def calculate_cost(self, sinr_map, lower_bound=4, upper_bound=20, use_squared=False):
        # 计算低于下限的惩罚
        below_penalty = np.maximum(lower_bound - sinr_map, 0)
        # 计算高于上限的惩罚
        above_penalty = np.maximum(sinr_map - upper_bound, 0)

        if use_squared:
            below_penalty = below_penalty ** 2
            above_penalty = above_penalty ** 2

        # 总惩罚
        total_cost = np.sum(below_penalty + above_penalty)
        # 或者计算平均惩罚
        average_cost = total_cost / sinr_map.size

        return average_cost

    def calculate_reward(self, sinr_matrix, power_consumption, power_weight=0.8):
        # 计算满足 SINR 约束的比例（正向激励）
        satisfied_mask = (sinr_matrix >= 4) & (sinr_matrix <= 20)
        satisfaction_ratio = np.mean(satisfied_mask)

        # 基础奖励：满足比例越高，奖励越高
        reward = satisfaction_ratio
        print(f'sinr奖励为{reward}')
        # 惩罚项：功率消耗（需归一化处理）
        normalized_power = power_consumption / 40  # 假设 MAX_POWER 是最大理论功率
        reward -= power_weight * normalized_power
        print(f'功率奖励为{ power_weight * normalized_power}')
        print(f'奖励为{reward}')
        return reward


def dbm_to_watt(dbm):
    return 10**(dbm/10)*1e-3
