import math
from symbol import continue_stmt

from scipy.interpolate import LinearNDInterpolator
import gymnasium as gym
from gymnasium.spaces import Discrete
from scipy.stats import rice
from configs.config_env import *
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal
import time
GRID_NOT_GENERATED_FLAG = -1
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
        self.sinr = None
        self.state = []
        self.density = None
        self.time_ = 0
        self.target = 0

        # 动作空间 (基站位置调整)
        self.action_space = self.action_space = Discrete(7)

        # 基站位置
        self.bs = np.array(bs_position)
        # 将存储的大尺度衰落转化为数组
        self.large_scale = np.array(self.calculate_large_scale())


    def reset(self, time_, seed = None,options = None):
        """重置环境"""
        self.state = []
        self.sinr = self.sinr_map()
        self.density = generate_distribution(time_)
        self.state.append(self.sinr)
        self.state.append(self.density)
        self.time_ = time_
        self.target = self.time_+40
        return self.state

    def step(self, action,test_flag=False):
        """环境状态更新"""
        # 更新基站位置

        action = action_space[action]
        #print(f'本次选择的动作为：{action}')
        self.tx_power[4] = action
        #self.tx_power[4] = np.clip(self.tx_power[4], 37, 43)  # 限制在基站发射功率内
        # 计算奖励 (以平均 SINR 为奖励)
        sinr_map = self.sinr_map()
        last_state = self.state
        self.state = []

        self.state.append(sinr_map)
        if test_flag == True:
            self.time_ += 5
        else:
            self.time_ += 5
        # 区域密度
        density_map = generate_distribution(self.time_)
        self.state.append(density_map)
        cost = self.calculate_cost(sinr_map,last_state[1])
        power_consumption = dbm_to_watt(self.tx_power[4])
        reward = self.calculate_reward(sinr_map,power_consumption,last_state[1])
        # 终止条件 (这里假设单步不终止，可根据实际需求调整)
        if (self.time_ == self.target) and test_flag == False:
            done = True
        elif self.time_ == 260 and test_flag == True:
            done = True
        else:
            done = False


        return self.state, reward, cost ,done, {}


    def sinr_map(self):

        sinr_values = self.calculate_sinr(x_sampled, y_sampled, z_sampled)
        # 创建一个更细的网格，覆盖整个200x200的区域（例如2000x2000网格）
        x_grid, y_grid = np.meshgrid(np.linspace(90, 140, 500), np.linspace(60, 90, 300))

        # 使用SciPy的griddata进行插值，采用'linear'插值方法
        interp = LinearNDInterpolator(tri, sinr_values)
        sinr_grid = interp((x_grid, y_grid))
        #sinr_grid = griddata((x_sampled, y_sampled), sinr_values, (x_grid, y_grid), method='linear')
        # 查找 NaN 值的位置
        nan_mask = np.isnan(sinr_grid)
        # 仅对NaN值的位置进行填充，使用 'nearest' 方法进行填充
        sinr_grid[nan_mask] = griddata((x_sampled, y_sampled), sinr_values, (x_grid[nan_mask], y_grid[nan_mask]),
                                       method='nearest')
        return sinr_grid   # 2000*2000


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

        return sinr_db

    def calculate_cost(self, sinr_map,density_map, lower_bound=4 ):
        # 计算低于下限的惩罚
        lower_bound = density_map*3
        below_penalty = np.maximum(lower_bound - sinr_map, 0)
        below_penalty = below_penalty * density_map
        # 计算高于上限的惩罚
        #above_penalty = np.maximum(sinr_map - upper_bound, 0)

        below_penalty = np.sqrt(below_penalty)

        return below_penalty

    def calculate_reward(self, sinr_matrix, power_consumption, density_map,power_weight=0.8):
        # 计算满足 SINR 约束的比例（正向激励）
        density = np.mean(density_map)  # 代表整体人员状况
        satisfied_mask = (sinr_matrix >= 4) & (sinr_matrix <= 40)  # 代表sinr状况
        weighted_satisfaction = np.mean(satisfied_mask * density_map)   # sinr平均状况
        #print(weighted_satisfaction)
        if density <0.5 and weighted_satisfaction<0.49:   # 人少 sinr差
            reward = 1.2 -0.3 * power_consumption + weighted_satisfaction*2
        elif density <0.5 and weighted_satisfaction>=0.49:    # 人少 sinr太好，惩罚功耗
            reward = -0.8 * power_consumption    # [-1,-2]
        elif density >=0.5 and weighted_satisfaction>=0.49:   # 人多 sinr太好，加权
            reward = -0.1 * math.sqrt(power_consumption)+weighted_satisfaction*6   #[0.5,]
        else:
            reward = -1/weighted_satisfaction  # [0,-4] 人多 sinr太差，根据sinr惩罚

        return reward


# 场地尺寸（国际足联标准足球场尺寸参考）
x_min_, x_max_ = 0, 50  #
y_min_, y_max_ = 0, 30  #
K = 5  # 保持8个高斯分量

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
def generate_distribution(t_minutes):     # 生成50*50范围内的密度分布
    xx, yy = np.meshgrid(np.linspace(x_min_, x_max_, 500),
                         np.linspace(y_min_, y_max_, 300))
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

    return density

def dbm_to_watt(dbm):
    return 10**(dbm/10)*1e-3
