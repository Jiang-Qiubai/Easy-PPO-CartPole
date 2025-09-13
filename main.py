import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 超参数设置 - 这些是影响训练效果的关键 knob，可以后续调整
class Config:
    learning_rate = 3e-4
    gamma = 0.99  # 折扣因子，衡量未来奖励的重要性
    lmbda = 0.95  # GAE 优势函数估计的参数
    eps_clip = 0.2  # PPO 中用于限制策略更新幅度的 clipping 参数
    K_epochs = 4  # 每次更新时，对同一批数据重复训练的轮数
    T_horizon = 20  # 每次收集多少步的数据后进行更新
    max_episodes = 3000  # 最多训练多少回合

# 构建神经网络：同时输出动作概率（Actor）和状态价值（Critic）
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        # 共享的特征提取层
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Actor 头部：输出每个动作的概率分布
        self.actor = nn.Linear(256, action_dim)
        # Critic 头部：输出一个标量，代表当前状态的价值
        self.critic = nn.Linear(256, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self):
        raise NotImplementedError
    
    # 通过网络获取动作的概率分布
    def pi(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.actor(x)
        prob = self.softmax(x)
        return prob
    
    # 通过网络获取状态的价值 V(s)
    def v(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        v = self.critic(x)
        return v

# PPO 智能体
class PPO:
    def __init__(self, state_dim, action_dim):
        self.config = Config()
        self.model = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # 用于存储一个批次的数据
        self.data = []
        
    # 将交互数据存入内存
    def put_data(self, transition):
        self.data.append(transition)
        
    # 清空当前批次的数据
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            
        # 将列表转换为 PyTorch 张量
        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float)
        a_batch = torch.tensor(np.array(a_lst))
        r_batch = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        prob_batch = torch.tensor(np.array(prob_a_lst), dtype=torch.float)
        done_batch = torch.tensor(np.array(done_lst), dtype=torch.float)
        
        self.data = []  # 清空数据
        return s_batch, a_batch, r_batch, s_prime_batch, prob_batch, done_batch
    
    # 核心：使用存储的数据更新网络
    def train(self):
        s, a, r, s_prime, prob_a, done = self.make_batch()
        
        # 计算优势函数和回报
        with torch.no_grad():
            # 计算 V(s) 和 V(s')
            v_s = self.model.v(s)
            v_s_prime = self.model.v(s_prime)
            # 计算 TD 残差：δ_t = r_t + γ * V(s_{t+1}) * done_mask - V(s_t)
            td_target = r + self.config.gamma * v_s_prime * done
            delta = td_target - v_s
            # 计算优势函数 Advantage (使用 GAELambda，简化版)
            advantage = delta.clone()
            for t in reversed(range(len(delta))):
                if t == len(delta) - 1:
                    advantage[t] = delta[t]
                else:
                    advantage[t] = delta[t] + self.config.gamma * self.config.lmbda * advantage[t+1]
        
        # 对旧数据迭代更新 K_epochs 次
        for _ in range(self.config.K_epochs):
            # 重新计算新策略下的动作概率和状态价值
            new_prob = self.model.pi(s).gather(1, a)  # 新策略下选择动作 a 的概率
            new_v = self.model.v(s)
            
            # 计算概率比：ratio = π_new(a|s) / π_old(a|s)
            ratio = torch.exp(torch.log(new_prob) - torch.log(prob_a))
            
            # PPO 的 Clipped Objective 核心公式
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()  # 策略损失
            
            # Critic 损失：让价值网络更准确地估计回报
            critic_loss = nn.MSELoss()(new_v, td_target.detach())
            
            # 总损失为两者之和，并加上一点熵奖励以鼓励探索
            entropy = -torch.sum(self.model.pi(s) * torch.log(self.model.pi(s)), dim=1).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 主训练函数
def main():
    # 创建环境
    env = gym.make("CartPole-v1", max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}. Action dimension: {action_dim}")
    
    # 创建智能体
    agent = PPO(state_dim, action_dim)
    
    score = 0.0  # 当前回合的总奖励
    episode = 0
    print_interval = 20  # 每20回合打印一次平均分数
    scores = []  # 记录每个回合的分数，用于绘图
    
    # 训练循环
    while episode < agent.config.max_episodes:
        # 重置环境，获取初始状态
        s, _ = env.reset()
        done = False
        
        while not done:
            for t in range(agent.config.T_horizon):
                # 1. 根据当前策略选择动作
                prob = agent.model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                action = m.sample().item()
                
                # 2. 在环境中执行动作
                s_prime, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 3. 存储交互数据 (s, a, r, s', prob_a, done)
                agent.put_data((s, action, r, s_prime, prob[action].item(), done))
                
                s = s_prime
                score += r
                
                if done:
                    break
                    
            # 4. 收集够 T_horizon 步数据（或回合结束），开始更新网络
            agent.train()
            
        # 一个回合结束
        episode += 1
        scores.append(score)
        score = 0.0
        
        # 定期打印训练进度
        if episode % print_interval == 0:
            avg_score = np.mean(scores[-print_interval:])
            print(f"Episode: {episode:4d}, Average Score: {avg_score:.2f}")
            # 如果平均分达到195（接近满分200），可以提前结束
            if avg_score > 195:
                print("Solved!")
                break
                
    # 训练结束，关闭环境
    env.close()
    
    # 绘制分数曲线
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('PPO Training Performance on CartPole-v1')
    plt.savefig('ppo_cartpole_score.png')
    plt.show()

    return agent, scores

def test_model(agent, episodes=5):
    test_env = gym.make('CartPole-v1', render_mode='human') # 使用 'human' 模式来渲染画面
    for ep in range(episodes):
        s, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad(): # 测试时不需要计算梯度
                prob = agent.model.pi(torch.from_numpy(s).float())
                action = torch.argmax(prob).item() # 直接选择概率最大的动作
            s, r, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += r
        print(f"Test Episode {ep+1}, Total Reward: {total_reward}")
    test_env.close()

if __name__ == '__main__':
    agent, scpres = main()
    test_model(agent)