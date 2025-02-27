from torch import optim

from configs.config_env import *
from models.sicmdp import *
from envs.basestation_env import *
from plot.plot_env import plot_reward, plot_env

"""
    获取基站和用户的位置
    这里首先采用固定的位置做测试
    user : (x,y)
    bs   : (x,y,z)
    build: (x,y,width,length,height)
    tree : (x,y,trunk_height, crown_height, radius)
"""

# 创建地形环境




if __name__ == "__main__":
    # 初始化环境和策略
    env = BaseStationEnv()
    state = env.reset()[2]
    state_dim = 6  # 对应preprocess_state提取的特征维度
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # 开始训练
    history = train(env, policy, optimizer, episodes=100)
    print(history)
    plot_reward(history)
    x = env.reset()
    plot_env(x[0],x[1],x[2])