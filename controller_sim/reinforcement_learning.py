#!~/anaconda3/envs/ros_env/bin/python

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float)

# Actor Net
# actor 输入状态state,输出最大Q的 action
class Actor(nn.Module):
    """
    Actor net work of ddpg framework

    Attributes:
        layer_input: nn.Linear linear fully connected
        layer_output: nn.Linear linear fully connected
    
    """
    def __init__(self, num_state,num_action, num_neuro):
        super(Actor, self).__init__() # 继承nn.module 的初始化

        # 定义神经网络层

        # 输入层: input: state 数量 -> output: 中间拟合神经元数    
        self.layer_input = nn.Linear(num_state, num_neuro) #线性全连接网络
        # 初始化输入层 网络参数
        nn.init.normal_(self.layer_input.weight, 0., 0.1)
        nn.init.constant_(self.layer_input.bias, 0.1)
        #self.layer_input.weight.data.normal_(0.0,0.1)
        #self.layer_input.bias.data.normal_(0.0,0.1)

        # 输出层: input: 中间拟合神经元数 -> output: action 数量;
        self.layer_output = nn.Linear(num_neuro, num_action) #线性全连接网络
        # 初始化输出层 网络参数
        nn.init.normal_(self.layer_output.weight, 0., 0.1)
        nn.init.constant_(self.layer_output.bias, 0.1)
        #self.layer_output.weight.data.normal_(0.0,0.1)
        #self.layer_output.bias.data.normal_(0.0,0.1)

    def forward(self, state):
        """
        actor 前向传递, 输入状态,输出动作

        Args:
            state: 输入状态

        Returns: 
            action: 输出动作, 使用 torch.tanh 激活,因此输出范围是[-1,1]
        """

        action = F.relu(self.layer_input(state)) # 输入层拟合后激活
        action = torch.tanh(self.layer_output(action)) # 上一层输出 放入输出层后激活
        return action # [-1,1]

# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):
    """
    Critic net work of ddpg framework

    Attributes:
        layer_input_s: nn.Linear linear fully connected
        layer_output_a: nn.Linear linear fully connected
    """
    def __init__(self, num_state,num_action, num_neuro):
        super(Critic, self).__init__()

        # 定义神经网络层
        # layer_input_s state -> num_neuro
        self.layer_input_s = nn.Linear(num_state, num_neuro)
        nn.init.normal_(self.layer_input_s.weight, 0., 0.1)
        nn.init.constant_(self.layer_input_s.bias, 0.1)

        # layer_input_a action -> num_neuro
        self.layer_input_a = nn.Linear(num_action, num_neuro)
        nn.init.normal_(self.layer_input_a.weight, 0., 0.1)
        nn.init.constant_(self.layer_input_a.bias, 0.1)

        # layer_output num_neuro -> 1 q value
        self.layer_output = nn.Linear(num_neuro, 1)
        #nn.init.normal_(self.layer_output.weight, 0.0, 0.1)
        #nn.init.constant_(self.layer_output.bias, 0.1)

    def forward(self, state, action):
        """
        critic 前向传递, 输入状态和动作, 输出动作的价值 Q 值

        Args:
            state: 输入状态
            action: 输入动作
            
        Returns: 
            q_value: 动作策略的q值, 使用 torch.relu 激活,因此输出范围是 [0,q_max]
        """
        state = self.layer_input_s(state)
        action = self.layer_input_a(action)
        q_value = self.layer_output(F.relu(state + action))
        return q_value # [0,q_max]

# DDPG Deep Deterministic Policy Gradient
class DDPG(object):
    """
    DDPG reinforcement learning framework

    only include soft replacement in this version

    Attributes:
        state_dim:
        action_dim:
        replacement:
        memory_size:
        gamma:
        lr_actor
        lr_critic
        batch_size



    """
    def __init__(self, 
                state_dim, 
                action_dim, 
                replacement, 
                memory_size= 1000, 
                gamma =0.9, 
                lr_actor = 0.001,
                lr_critic = 0.001,
                batch_size = 40):
        """
        init

        Args:
            state_dim: state dimension
            action_dim: action dimension
            replacement: replacement, only avilible for soft replacement, dict(name='soft', tau=0.005),
            memory_size: memory of experience
            gamma: 衰减率
            lr_actor: 学习率 actor
            lr_critic: 学习率 critic
            batch_size: 
        """
        super(DDPG, self).__init__() # 继承父对象的初始化方法
        # 初始化参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replacement = replacement
        self.memory_size = memory_size
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size

        # 采样周期数,迭代次数
        self.count_iter = 0

        # [state, state_, action, reward]
        self.memory = np.zeros((self.memory_size,state_dim + state_dim + action_dim +1))

        # actor 
        self.actor_eval = Actor(state_dim,action_dim,num_neuro=30) # 更新网络
        self.actor_target = Actor(state_dim,action_dim,num_neuro=30) # 只用来生成目标的网络

        # critic
        self.critic_eval = Critic(state_dim,action_dim,num_neuro=30)
        self.critic_target = Critic(state_dim,action_dim,num_neuro=30)

        # optimizer 
        self.opt_a = torch.optim.Adam(self.actor_eval.parameters(),lr=lr_actor) 
        self.opt_c = torch.optim.Adam(self.critic_eval.parameters(),lr=lr_critic) 
        # loss function
        self.closs = nn.MSELoss()
        # self.aloss = torch.mean()

    def sample(self):
        # 取 memory_size 内 batch_size 个随机数
        indices = np.random.choice(self.memory_size, size = self.batch_size)
        # 根据随机索引生成训练 batch
        return self.memory[indices,:]

    def choose_action(self, state):
        state = torch.FloatTensor(state) # 转化为 tensor
        action = self.actor_eval(state) # [-1,1]
        return action.detach().numpy() # 生成的action 不参与网络训练, 因此从图上detach

    def learn(self):
        # soft update target net
        # if self.replacement['name'] == 'soft':
        tau = self.replacement['tau'] # 更新率
        #通过 eval 更新 target
        # target_weight <- target_weight + tau * [eval_weight - weight_target]
        # target_weight <- target_weight *(1.0-tau) + tau * eval_weight
        # .name_childrn 生成可迭代的列表 [对象名, 对象]
        a_layers = self.actor_target.named_children()
        c_layers = self.critic_target.named_children()
        for alayer in a_layers:
            alayer[1].weight.data.mul_((1-tau))
            alayer[1].weight.data.add_(tau * self.actor_eval.state_dict()[alayer[0]+'.weight'])
            alayer[1].bias.data.mul_((1-tau))
            alayer[1].bias.data.add_(tau * self.actor_eval.state_dict()[alayer[0]+'.bias'])

        for clayer in c_layers:
            clayer[1].weight.data.mul_((1-tau))
            clayer[1].weight.data.add_(tau * self.critic_eval.state_dict()[clayer[0]+'.weight'])
            clayer[1].bias.data.mul_((1-tau))
            clayer[1].bias.data.add_(tau * self.critic_eval.state_dict()[clayer[0]+'.bias'])

        batch = self.sample()
        # memory: [state, state_, action, reward]
        batch_s = torch.FloatTensor(batch[:,: self.state_dim])
        batch_s_= torch.FloatTensor(batch[:,self.state_dim : self.state_dim+self.state_dim])
        batch_a = torch.FloatTensor(batch[:,self.state_dim+self.state_dim : self.state_dim+self.state_dim+self.action_dim])
        batch_r = torch.FloatTensor(batch[:,self.state_dim+self.state_dim+self.action_dim:self.state_dim+self.state_dim+self.action_dim+1])

        # 训练Actor eval
        # batch_a 是加了噪声的, 不是由actor得到的
        action = self.actor_eval(batch_s)
        q_eval = self.critic_eval(batch_s,action)
        # 用最大增加梯度的q(本采样周期的) 来训练网络,即 求 -q 的最大减小梯度
        actor_loss = - torch.mean(q_eval)
        self.opt_a.zero_grad() # 清空梯度缓存
        actor_loss.backward(retain_graph=True)
        self.opt_a.step()

        # 训练critic eval

        # 根据 s_ 计算 下一采样周期 q_ 和 a_ 用于计算 q_target (所以要使用 target_net)
        a_target_ = self.actor_target(batch_s_) # 下一周期的a
        q_target_ = self.critic_target(batch_s_,a_target_)
        q_target = batch_r + self.gamma * q_target_
        q_eval = self.critic_eval(batch_s,batch_a)
        td_error = self.closs(q_target,q_eval)
        self.opt_c.zero_grad()
        td_error.backward()
        self.opt_c.step()

    def store_memory(self, s,s_,a,r):
        # memory: [state, state_, action, reward]
        memory_index = np.hstack((s,s_,a,r))
        index = self.count_iter % self.memory_size 
        self.memory[index, :] = memory_index
        self.count_iter +=1

# test ddpg using gym

if __name__ == '__main__':
    import gym
    import time
    # hyper parameters
    VAR = 2  #exploration
    MAX_EPISODES = 500
    MAX_EP_STEPS = 200
    MEMORY_CAPACITY = 10000
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
    ][0]

    ENV_NAME = 'Pendulum-v1'
    RENDER = False

    env = gym.make(ENV_NAME)
    env = env.unwrapped

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                replacement=REPLACEMENT,
                memory_size=MEMORY_CAPACITY)    

    
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            t1 = time.time()

            a = ddpg.choose_action(s) # [-1,1]
            
            a =np.random.normal(a, VAR) # 加入噪声
            a =torch.tanh(torch.FloatTensor(a)) # 用 tanh 重新 map 到 [-1,1]
            a = a_bound * a.numpy() # 
            #a = np.clip(np.random.normal(a, VAR), -2, 2)
            s_,r,done,_,info = env.step(a)

            ddpg.store_memory(s, s_, a, r)
            if ddpg.count_iter > MEMORY_CAPACITY:
                VAR *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_

            t2 = time.time()
            
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                print('calculate_time: ', t2 - t1) #0.007s
                if ep_reward > -300: RENDER = True
                break
    
    






