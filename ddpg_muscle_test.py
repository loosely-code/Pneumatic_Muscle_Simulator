import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

import muscle_sim.muscle_env as muscle_env
import muscle_sim.signal_gen as signal_gen
from controller_sim.reinforcement_learning import DDPG

if __name__ == '__main__':
    MAX_EPISODES = 100
    MEMORY_CAPACITY = 5000
    VAR =2
    REPLACEMENT = [
        dict(name='soft', tau=0.01),
        dict(name='hard', rep_iter=600)
    ][0]

    # define the simulation parameters
    t_total = 5.0  # 5s in realtime
    t_sample = 0.01
    N_iter = int(t_total // t_sample) 
    load0 = 1000  # 外载荷
    p_in = 550  # 输入气压
    mass = 0.0002 # 0.0002*10^3 = 0.2kg

    # initialization

    # initialize the environment
    env = muscle_env.Three_element_env(
        T_sample= t_sample,
        Load_0= load0,
        Mass= mass
    )

    # initialize signal 
    sig_target = signal_gen.signal_sin(
        Amp= 10,
        Period= 2.5,
        Center= 10,
        T_sample=t_sample
    )

    # initialize controller
    ddpg = DDPG(state_dim=3,
    action_dim=1,
    replacement= REPLACEMENT,
    memory_size= MEMORY_CAPACITY
    )

    # initialize plot memory
    plotMemory_state = np.zeros([N_iter, 2])
    plotMemory_time = np.zeros([N_iter, 1])
    plotMemory_ep_reward = np.zeros([MAX_EPISODES, 1])
    plotMemory_target = np.zeros([N_iter, 1])
    
    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

    ax[0].set_title("Displacement")
    ax[1].set_title("Velocity")
    ax[2].set_title("EP_Reward")

    plt.ion()
    for iter_episode in range (MAX_EPISODES):
        state = env.reset()
        pos_d = sig_target.reset()
        error = pos_d - state[0]
        state_rl = np.hstack((state,[error]))
        ep_reward = 0
        reward = 0

        for iter_k in range(N_iter):
            plotMemory_time[iter_k] = env.get_t()
            plotMemory_state[iter_k,0] = state[0]
            plotMemory_state[iter_k,1] = (state[0] - state[1]) / t_sample
            plotMemory_target[iter_k] = pos_d

            action = ddpg.choose_action(state_rl)
            action = np.random.normal(action,VAR)
            action = torch.tanh(torch.FloatTensor(action))
            pressure_input = 350 + action.numpy() *200

            state_next = env.step(U_k=pressure_input, Load_k=load0)
            pos_d = sig_target.step()
            error_next = pos_d - state_next[0]
            state_rl_next = np.hstack((state_next,[error_next]))

            reward = - error **2 - 0.2 * (error_next -error)**2
            if abs(error) < 0.1:
                reward +=20
            if abs(error) < 0.01:
                reward +=50

            ddpg.store_memory(state_rl, state_rl_next,action,reward/100)

            if ddpg.count_iter > MEMORY_CAPACITY:
                VAR *= 0.995
                ddpg.learn()

            state = state_next
            state_rl = state_rl_next
            error = error_next
            ep_reward += reward

            if iter_k == N_iter -1:
                plotMemory_ep_reward[iter_episode] = ep_reward
                print('Episode:', iter_episode, ' Episode reward: %i' % int(ep_reward), 'VAR: %.2f' % VAR, )
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()
                ax[0].plot(plotMemory_time,plotMemory_target, "--")
                ax[0].plot(plotMemory_time,plotMemory_state[:,0])
                ax[1].plot(plotMemory_time,plotMemory_state[:,1])
                ax[2].plot(range(MAX_EPISODES),plotMemory_ep_reward)
                plt.pause(0.1)
                

    plt.ioff()
    plt.show()        

