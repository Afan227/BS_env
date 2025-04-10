from torch import optim
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

    state_dim = 8  # 对应preprocess_state提取的特征维度
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim)
    # 查看第一个全连接层的权重
    print("第一层权重（部分）：\n", policy.fc[0].weight.data[:2, :4])  # 打印前两行、前四列

    # 查看最后一层的偏置
    print("最后一层偏置：\n", policy.fc[4].bias.data)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # 开始训练
    history_r,history_c = train(env, policy, optimizer, episodes=200)
    print(history_r)
    plot_reward(history_r,mode='reward')
    plot_reward(history_c,mode='cost')
    x = env.reset(0)
    x_grid, y_grid = np.meshgrid(np.linspace(0, 200, 2000), np.linspace(0, 200, 2000))
    target_array = np.zeros((2000, 2000))

    # 将x0的数据放入目标数组的对应位置
    target_array[600:900,900:1400 ] = x[0]
    plot_env(x_grid,y_grid,target_array)