import torch
import torch.nn as nn
import time
import numpy as np

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

    def forward(self, x):
        return self.fc(x)

def preprocess_state(state):
    """将大型网格转换为全连接网络可处理的特征向量"""
    x_grid, y_grid, sinr_grid = state

    # ================== 方案1：统计特征提取 ==================
    # 处理SINR网格（核心特征）
    sinr_stats = [
        np.mean(sinr_grid),  # 均值
        np.max(sinr_grid),  # 最大值
        np.min(sinr_grid),  # 最小值
        np.std(sinr_grid),  # 标准差
        np.percentile(sinr_grid, 25),  # 25%分位数
        np.percentile(sinr_grid, 75)  # 75%分位数
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
    block_features = block_pooling(sinr_grid, block_size=200)  # 降为10×10=100维

    # ================== 特征组合 ==================
    # 方法选择建议：
    # - 实时性要求高：优先方案1（约10+维）
    # - 需要空间信息：用方案2（200×200分块得到100维）
    # - 折中方案：组合方案1+轻量方案2（如500×500分块）

    # 示例组合特征（按需选择）：
    features = sinr_stats

    return torch.tensor(features, dtype=torch.float32)

def train(env, policy, optimizer, episodes=100, gamma=0.99):
    print('训练开始')
    rewards_history = []
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        costs = []
        done = False
        i = 0
        while not done:
            # 预处理状态

            state_tensor = preprocess_state(state)

            # 选择动作
            action_probs = policy(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            print(f'本次的采样空间概率为{action_probs}')
            action = dist.sample()

            log_prob = dist.log_prob(action)  #log_prob = log(action_probs[action])

            # 执行动作
            next_state, reward, cost, done, _ = env.step(action)

            # 保存数据
            log_probs.append(log_prob)
            rewards.append(reward)
            costs.append(cost)
            #print(reward)
            state = next_state
            i+=1
            if i == 50:
                done = True
                print('当前episode完成')

        # 计算折扣回报
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        # 归一化回报（降低方差）
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # 计算损失
        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()

        # 更新策略
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # 记录训练过程
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    return rewards_history

