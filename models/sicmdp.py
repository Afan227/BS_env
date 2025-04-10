import torch
import torch.nn as nn
import time
import numpy as np
from configs.config_env import *


class PolicyNetwork(nn.Module):
    """策略网络：输入状态特征，输出动作概率"""

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        #self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.fc[-2]:  # 输出层前一层
                    nn.init.orthogonal_(m.weight, gain=0.1)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.fc(x)

def preprocess_state(state):
    """将大型网格转换为全连接网络可处理的特征向量"""
    sinr_grid,density = state[0],state[1]
    # x_min, x_max = 90, 140  # 假设子区域中心为(100,100)
    # y_min, y_max = 40, 90
    # sinr_grid = sinr_grid[y_min*10:y_max*10,x_min*10: x_max*10]
    # ================== 方案1：统计特征提取 ==================
    # 处理SINR网格（核心特征）
    sinr_stats = [
        np.mean(sinr_grid),  # 均值
        #np.max(sinr_grid),  # 最大值
        #np.min(sinr_grid),  # 最小值
        #np.std(sinr_grid),  # 标准差
        np.percentile(sinr_grid, 25),  # 25%分位数
        np.percentile(sinr_grid, 75)  # 75%分位数
    ]
    density_stats = [
        np.mean(density),  # 均值
        np.max(density),  # 最大值
        #np.min(density),  # 最小值
        np.std(density),  # 标准差
        np.percentile(density, 25),  # 25%分位数
        np.percentile(density, 75)  # 75%分位数
    ]

    # ================== 方案2：分块池化降维 ==================
    def block_pooling(grid, block_size=100, pool_func=np.max):
        """将网格划分为块并池化"""
        h, w = grid.shape
        return [
            pool_func(grid[i:i + block_size, j:j + block_size])
            for i in range(0, h, block_size)
            for j in range(0, w, block_size)
        ]

    # 生成分块特征（示例使用最大池化）
    #block_features = block_pooling(sinr_grid, block_size=200)  # 降为10×10=100维

    # ================== 特征组合 ==================
    # 方法选择建议：
    # - 实时性要求高：优先方案1（约10+维）
    # - 需要空间信息：用方案2（200×200分块得到100维）
    # - 折中方案：组合方案1+轻量方案2（如500×500分块）

    # 示例组合特征（按需选择）：
    features = sinr_stats + density_stats


    return torch.tensor(features, dtype=torch.float32)

def s_k(s_k_costs,grid_rows,grid_cols):
    V_c_hat_Array = np.zeros((100, 100))
    for costs in s_k_costs:
        V_pi_c = np.zeros((100, 100))
        for t in reversed(range(len(costs))):
            costs[t] = costs[t][grid_rows, grid_cols]
            V_pi_c = costs[t] + 0.9 * V_pi_c
        V_c_hat_Array += V_pi_c
    V_c_hat_Array = (V_c_hat_Array/num_traj)* LR / (1 + LR * kappa)
    return torch.exp(torch.tensor(V_c_hat_Array)).flatten().to(torch.float32)

VAL_E = 50
ROU_0 = 2
ROU_SITA = 5
M0 = ROU_0 / VAL_E
V = ROU_SITA / M0
kappa = 0.3
L = 1 / (1 + LR* kappa)
num_traj = 50
def train(env, policy, optimizer, episodes=100, gamma=0.9):
    print('训练开始')
    rewards_history = []
    costs_history = []

    lamda_n = torch.ones(10000) * M0

    for episode in range(episodes):
        start_time = time.time()
        policy_loss_all = 0
        s_k_cost = []
        # 生成行和列的等距索引（100个点）
        num_points_per_dim = int(np.sqrt(10000))  # 100
        rows = np.linspace(0, 499, num=num_points_per_dim, dtype=int)
        cols = np.linspace(0, 299, num=num_points_per_dim, dtype=int)

        # 构建坐标网格
        grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
        for i in range(num_traj):
            values = np.arange(0, 220 + 5, 5)
            weights = np.where((values <= 20) | (values >= 190), 5, 1)
            probabilities = weights / weights.sum()

            time_start = np.random.choice(values, size=1, p=probabilities)  # 抽 10 个样本
            #time_start = 0
            #time_start = np.random.choice(range(0, 220,5), 1)[0]
            state = env.reset(time_start)
            log_probs = []
            rewards = []
            costs = []
            done = False
            while not done:
                # 预处理状态

                state_tensor = preprocess_state(state)

                # 选择动作
                action_probs = policy(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                #print(f'本次的采样空间概率为{action_probs}')
                action = dist.sample()
                #print(action)
                log_prob = dist.log_prob(action)  #log_prob = log(action_probs[action])

                # 执行动作
                next_state, reward, cost, done, _ = env.step(action,test_flag = False)

                # 保存数据
                log_probs.append(log_prob)
                rewards.append(reward)
                costs.append(cost)
                #print(reward)
                state = next_state
                if done == True:
                    print('当前episode完成')

            # 计算折扣回报
            discounted_rewards = []
            discounted_costs = []


            # 提取采样点数据
            def sample_matrix(matrix, grid_rows, grid_cols,lamda):
                matrix = np.array(matrix[grid_cols,grid_rows]).flatten()
                max_cost = np.max(matrix)
                intergra = np.dot(matrix , lamda)* VAL_E/10000
                return intergra,max_cost

            # 计算轨迹折扣回报和折扣代价
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
            Cost = 0
            for cost in reversed(costs):
                Cost = cost + gamma * Cost
                discounted_costs.insert(0, Cost)

            results = np.array([sample_matrix(m, grid_rows, grid_cols, lamda_n) for m in discounted_costs])
            sampled_list = [r[0] for r in results]
            s_k_cost.append(discounted_costs)
            # 归一化回报（降低方差）
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = discounted_rewards - 0.8*torch.tensor(sampled_list)

            # 计算损失
            policy_loss = []
            for log_prob, R in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * R)

            policy_loss_all += torch.stack(policy_loss).sum()/num_traj
        #print(f'{policy_loss}')
        #不是，我指的是两条轨迹，一个轨迹的reward全是证书，一条轨迹的奖励全是负数
        # 更新策略
        policy_loss_all = policy_loss_all/num_traj
        optimizer.zero_grad()
        policy_loss_all.backward()
        optimizer.step()

        # 测度更新
        s_k_ = s_k(s_k_cost,grid_cols,grid_rows)

        lamda_n_little = lamda_n / M0
        lamda_n_l = torch.pow(lamda_n_little, L)
        #print(torch.dot(s_k_.flatten().to(torch.float32), lamda_n_l) / 10000)
        intergration = VAL_E * torch.dot(s_k_.flatten().to(torch.float32), lamda_n_l) / 10000
        #print(intergration)
        lamda_n = min(V / intergration, 1) * s_k_ * lamda_n_l * M0
        # LAMMDA = np.sum(lamda_n)
        #print(lamda_n)
        end_time = time.time()
        print(f'测试时间为{end_time - start_time}')
        total_reward,total_cost = test_episode(env,policy)
        rewards_history.append(total_reward)
        costs_history.append(total_cost)
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
        print(f"Episode {episode}, Total Max_cost: {total_cost:.2f}")

    return rewards_history,costs_history

def test_episode(env,policy):
    start_time = time.time()
    state = env.reset(0)
    log_probs = []
    rewards = []
    costs = []
    done = False

    while not done:
        # 预处理状态
        state_tensor = preprocess_state(state)

        # 选择动作
        action_probs = policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        # print(f'本次的采样空间概率为{action_probs}')
        action = dist.sample()
        print(action,end='')

        log_prob = dist.log_prob(action)  # log_prob = log(action_probs[action])

        # 执行动作
        next_state, reward, cost, done, _ = env.step(action,test_flag=True)

        # 保存数据
        log_probs.append(log_prob)
        rewards.append(reward)
        costs.append(cost)
        # print(reward)
        state = next_state
        if done == True:
            print('当前episode完成')

    # 生成行和列的等距索引（100个点）
    num_points_per_dim = int(np.sqrt(10000))  # 100
    rows = np.linspace(0, 499, num=num_points_per_dim, dtype=int)
    cols = np.linspace(0, 299, num=num_points_per_dim, dtype=int)

    # 构建坐标网格
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')

    # 提取采样点数据
    def sample_matrix(matrix, grid_rows, grid_cols):
        matrix = np.array(matrix[grid_rows, grid_cols]).flatten()
        max_cost = np.max(matrix)
        return max_cost

    V_pi_c = np.zeros((100, 100))
    for t in reversed(range(len(costs))):
        costs[t] = costs[t][grid_cols, grid_rows]
        V_pi_c = costs[t] + 0.9 * V_pi_c
    max_v = np.max(V_pi_c)
    total_reward = sum(rewards)
    end_time = time.time()
    print(f'测试时间为{end_time-start_time}')
    return total_reward,max_v